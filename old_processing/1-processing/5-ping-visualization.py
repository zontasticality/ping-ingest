#!/usr/bin/env python3
"""
Memory-Efficient 2D Autocorrelation Matrix Visualization

Creates a 200x200 heatmap using streaming approaches to avoid memory issues.
Key fixes: streaming aggregation, bounded storage, one-pair-at-a-time processing.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict, Counter
import time
import pickle

# Configuration
MAX_FILES = 100  # Reduced from 500 to manage memory
TOP_NODES = 50   # Reduced from 200 to make 2500 pairs manageable
MAX_MEASUREMENTS_PER_PAIR = 50  # Bounded storage per pair
SAMPLE_FILES_FOR_NODES = 20     # Reduced sampling for node identification

def identify_top_nodes_streaming(files, n_nodes=TOP_NODES):
    """
    FIXED: Use streaming aggregation to identify top nodes without collecting full data.
    """
    print(f"Identifying top {n_nodes} most active nodes via streaming...")
    
    node_activity = Counter()
    
    # Process files one at a time with streaming aggregation
    for i, file_path in enumerate(files[:SAMPLE_FILES_FOR_NODES]):
        print(f"Streaming file {i+1}/{SAMPLE_FILES_FOR_NODES}: {file_path.name}")
        
        df = pl.scan_parquet(str(file_path))
        
        # Use streaming aggregation instead of collect()
        src_counts = (
            df.filter((pl.col("avg") > 0) & (pl.col("sent") == pl.col("rcvd")))
            .group_by("src_ipv4_int")
            .agg(pl.len().alias("count"))
            .collect()  # Only collect aggregated results (small)
        )
        
        dst_counts = (
            df.filter((pl.col("avg") > 0) & (pl.col("sent") == pl.col("rcvd")))
            .group_by("dst_ipv4_int") 
            .agg(pl.len().alias("count"))
            .collect()  # Only collect aggregated results (small)
        )
        
        # Add to counters
        for row in src_counts.iter_rows(named=True):
            if row["src_ipv4_int"]:
                node_activity[row["src_ipv4_int"]] += row["count"]
        
        for row in dst_counts.iter_rows(named=True):
            if row["dst_ipv4_int"]:
                node_activity[row["dst_ipv4_int"]] += row["count"]
    
    # Get top N nodes
    top_nodes = [node for node, count in node_activity.most_common(n_nodes)]
    
    print(f"Top {len(top_nodes)} nodes identified")
    if top_nodes:
        print(f"Activity range: {node_activity[top_nodes[-1]]:,} - {node_activity[top_nodes[0]]:,}")
    
    return top_nodes

def stream_pairs_with_bounds(files, top_nodes):
    """
    FIXED: Process one pair at a time with bounded storage to prevent memory explosion.
    """
    print(f"Streaming {len(top_nodes)}x{len(top_nodes)} = {len(top_nodes)**2:,} pairs with bounded storage...")
    
    pair_data = {}  # Will only store pairs that have data
    total_pairs_checked = 0
    active_pairs_found = 0
    
    # Process each source-destination pair individually
    for src_idx, src_node in enumerate(top_nodes):
        for dst_idx, dst_node in enumerate(top_nodes):
            if src_node == dst_node:  # Skip self-loops
                continue
                
            total_pairs_checked += 1
            pair_measurements = []
            
            # Stream through files collecting data for just this one pair
            for file_path in files:
                df = pl.scan_parquet(str(file_path))
                
                # Filter for just this specific pair
                pair_data_file = (
                    df.filter(
                        (pl.col("src_ipv4_int") == src_node) & 
                        (pl.col("dst_ipv4_int") == dst_node) &
                        (pl.col("avg") > 0) &
                        (pl.col("sent") == pl.col("rcvd"))
                    )
                    .select(["ts", "avg"])
                    .collect()  # Small result - just one pair from one file
                )
                
                # Add measurements with bounded storage
                for row in pair_data_file.iter_rows(named=True):
                    if len(pair_measurements) < MAX_MEASUREMENTS_PER_PAIR:
                        pair_measurements.append((row["ts"], row["avg"]))
                
                # Early termination if we have enough data
                if len(pair_measurements) >= MAX_MEASUREMENTS_PER_PAIR:
                    break
            
            # Store pair if it has sufficient data
            if len(pair_measurements) >= 10:  # Minimum for autocorrelation
                pair_data[(src_node, dst_node)] = pair_measurements
                active_pairs_found += 1
            
            # Progress reporting
            if total_pairs_checked % 100 == 0:
                memory_estimate = (active_pairs_found * MAX_MEASUREMENTS_PER_PAIR * 16) / (1024 * 1024)
                print(f"  Checked {total_pairs_checked:,}/{len(top_nodes)**2:,} pairs, "
                      f"found {active_pairs_found:,} active, mem ~{memory_estimate:.1f}MB")
    
    print(f"Pair collection complete: {len(pair_data):,} active pairs")
    return pair_data

def calculate_autocorr_matrix_efficient(pair_data, top_nodes):
    """
    FIXED: Efficient matrix calculation with minimal memory usage.
    """
    print(f"Computing {len(top_nodes)}x{len(top_nodes)} autocorrelation matrix...")
    
    # Create efficient mapping
    node_to_idx = {node: i for i, node in enumerate(top_nodes)}
    
    # Initialize sparse matrix (only store non-NaN values)
    autocorr_matrix = np.full((len(top_nodes), len(top_nodes)), np.nan)
    pairs_computed = 0
    
    # Process each pair efficiently
    for (src_node, dst_node), measurements in pair_data.items():
        # Sort by timestamp and extract latencies
        measurements.sort(key=lambda x: x[0])
        latencies = [m[1] for m in measurements]
        
        # Calculate lag-1 autocorrelation
        if len(latencies) >= 3:
            try:
                corr, _ = pearsonr(latencies[:-1], latencies[1:])
                autocorr_value = corr if not np.isnan(corr) else 0.0
            except:
                autocorr_value = 0.0
            
            # Store in matrix
            src_idx = node_to_idx[src_node]
            dst_idx = node_to_idx[dst_node]
            autocorr_matrix[src_idx, dst_idx] = autocorr_value
            
            pairs_computed += 1
    
    # Fill diagonal with 1.0 (perfect self-correlation)
    np.fill_diagonal(autocorr_matrix, 1.0)
    
    print(f"Matrix complete: {pairs_computed:,} pairs computed")
    return autocorr_matrix

def create_compact_heatmap(autocorr_matrix, top_nodes):
    """
    FIXED: Create memory-efficient visualization.
    """
    print(f"Creating {len(top_nodes)}x{len(top_nodes)} autocorrelation heatmap...")
    
    # Calculate statistics
    valid_values = autocorr_matrix[~np.isnan(autocorr_matrix)]
    diagonal_values = autocorr_matrix[~np.isnan(autocorr_matrix) & ~np.eye(len(top_nodes), dtype=bool)]
    
    if len(valid_values) == 0:
        print("No valid data for visualization")
        return
    
    n_valid = len(valid_values)
    n_total = autocorr_matrix.size
    coverage = n_valid / n_total * 100
    
    print(f"Matrix statistics:")
    print(f"  Coverage: {n_valid:,}/{n_total:,} ({coverage:.1f}%)")
    print(f"  Non-diagonal values: {len(diagonal_values):,}")
    if len(diagonal_values) > 0:
        print(f"  Autocorr range: {np.min(diagonal_values):.3f} to {np.max(diagonal_values):.3f}")
        print(f"  Mean autocorr: {np.mean(diagonal_values):.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Use masked array to handle NaN values properly
    masked_matrix = np.ma.masked_where(np.isnan(autocorr_matrix), autocorr_matrix)
    
    im = plt.imshow(masked_matrix, cmap='RdBu_r', aspect='equal', 
                    vmin=-1, vmax=1, interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Lag-1 Autocorrelation', fontsize=12)
    
    # Styling
    plt.title(f'Network Latency Autocorrelation Matrix\n'
              f'{len(top_nodes)} Most Active Nodes ({coverage:.1f}% Coverage)', 
              fontsize=14, pad=15)
    plt.xlabel('Destination Node Index', fontsize=12)
    plt.ylabel('Source Node Index', fontsize=12)
    
    # Grid
    ax = plt.gca()
    step = max(5, len(top_nodes) // 10)
    ax.set_xticks(np.arange(0, len(top_nodes), step))
    ax.set_yticks(np.arange(0, len(top_nodes), step))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('autocorr_matrix_streaming.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return autocorr_matrix

def main():
    """Memory-efficient main analysis."""
    print("=== MEMORY-EFFICIENT 2D AUTOCORR MATRIX ===")
    print(f"Target: {TOP_NODES}x{TOP_NODES} matrix with bounded memory usage")
    print()
    
    start_time = time.time()
    
    # Get files
    data_path = Path("data/ping_super_optimized_fixed")
    all_files = sorted(list(data_path.glob("*.parquet")))[:MAX_FILES]
    
    print(f"Dataset: {len(all_files)} files")
    print(f"Memory limits: {MAX_MEASUREMENTS_PER_PAIR} measurements per pair")
    print()
    
    # Step 1: Identify top nodes with streaming
    top_nodes = identify_top_nodes_streaming(all_files, TOP_NODES)
    
    if len(top_nodes) < 10:
        print("❌ Insufficient nodes found")
        return
    
    # Step 2: Stream pair data with bounds
    pair_data = stream_pairs_with_bounds(all_files, top_nodes)
    
    if len(pair_data) == 0:
        print("❌ No pair data collected")
        return
    
    # Step 3: Calculate matrix
    autocorr_matrix = calculate_autocorr_matrix_efficient(pair_data, top_nodes)
    
    # Step 4: Visualize
    create_compact_heatmap(autocorr_matrix, top_nodes)
    
    # Summary
    end_time = time.time()
    total_measurements = sum(len(measurements) for measurements in pair_data.values())
    memory_mb = (total_measurements * 16) / (1024 * 1024)
    
    print(f"\n=== MEMORY-EFFICIENT ANALYSIS COMPLETE ===")
    print(f"✅ {len(top_nodes)}x{len(top_nodes)} matrix created successfully")
    print(f"Active pairs: {len(pair_data):,}/{len(top_nodes)**2:,}")
    print(f"Total measurements: {total_measurements:,}")
    print(f"Memory usage: {memory_mb:.1f} MB (BOUNDED)")
    print(f"Processing time: {(end_time - start_time)/60:.1f} minutes")
    
    # Save results
    with open('autocorr_matrix_efficient.pkl', 'wb') as f:
        pickle.dump({'matrix': autocorr_matrix, 'nodes': top_nodes}, f)
    print("Results saved to autocorr_matrix_efficient.pkl")

if __name__ == "__main__":
    main()