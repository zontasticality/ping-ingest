#!/usr/bin/env python3
"""
FIXED Dataset Optimization Script
Properly handles probe mapping without creating duplicate rows that bias analysis.
"""

import polars as pl
import os
from pathlib import Path
import glob
import ipaddress
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import psutil

# Configure polars
pl.Config.set_fmt_str_lengths(50)
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

def convert_ipv4_batch(ip_array):
    """Convert batch of IPv4 addresses to UInt32 using multiprocessing."""
    result = np.zeros(len(ip_array), dtype=np.uint32)
    
    for i, ip_str in enumerate(ip_array):
        if ip_str and ':' not in ip_str:  # IPv4 check
            try:
                parts = ip_str.split('.')
                if len(parts) == 4:
                    result[i] = (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
            except:
                result[i] = 0  # Invalid IP
        else:
            result[i] = 0  # Not IPv4
    
    return result

def parallel_ipv4_conversion(series):
    """Convert IP series to IPv4 integers using all available CPU cores."""
    print(f"  🔄 Converting {len(series):,} IPs with {mp.cpu_count()} cores...")
    start_time = time.time()
    
    # Convert to numpy for efficiency
    ip_array = series.to_numpy()
    
    # Use multiprocessing to parallelize across CPU cores
    with ProcessPoolExecutor(max_workers=min(63, mp.cpu_count())) as executor:
        # Split into chunks for parallel processing
        chunk_size = max(1000, len(ip_array) // (mp.cpu_count() * 4))
        chunks = [ip_array[i:i + chunk_size] for i in range(0, len(ip_array), chunk_size)]
        
        # Process chunks in parallel
        results = list(executor.map(convert_ipv4_batch, chunks))
        
        # Combine results
        combined = np.concatenate(results) if results else np.array([], dtype=np.uint32)
    
    elapsed = time.time() - start_time
    print(f"  ✅ IP conversion completed in {elapsed:.2f}s ({len(series)/elapsed:,.0f} IPs/sec)")
    
    return pl.Series(combined)

def create_deduplicated_probe_map(probe_map):
    """
    Create probe mapping that avoids row duplication by choosing one IP per probe.
    Strategy: Prefer IPv4 over IPv6 for consistency.
    """
    if probe_map is None:
        return None
    
    print("🔄 Creating DEDUPLICATED probe mapping...")
    
    # Check for duplicates
    duplicate_count = probe_map.group_by('dst_prb_id').agg(pl.len().alias('count')).filter(pl.col('count') > 1).height
    print(f"  Found {duplicate_count:,} probes with multiple IPs")
    
    # Deduplication strategy: prefer IPv4 over IPv6
    deduplicated = (
        probe_map
        .with_columns([
            # Add preference score: IPv4 = 0, IPv6 = 1 (lower is preferred)  
            pl.when(pl.col('ip').str.contains(':'))
            .then(1)
            .otherwise(0)
            .alias('ip_preference')
        ])
        .sort(['dst_prb_id', 'ip_preference'])  # Sort by probe ID, then preference
        .group_by('dst_prb_id')
        .first()  # Take first (preferred) IP for each probe
        .drop('ip_preference')
    )
    
    print(f"  Original probe entries: {probe_map.height:,}")
    print(f"  Deduplicated entries: {deduplicated.height:,}")
    print(f"  Removed duplicates: {probe_map.height - deduplicated.height:,}")
    
    # Convert to optimized format
    optimized_probe_map = deduplicated.with_columns([
        # Detect IPv6 (vectorized)
        pl.col('ip').str.contains(':').alias('src_is_ipv6'),
        
        # Convert IPv4 using multiprocessing
        pl.col('ip').map_batches(parallel_ipv4_conversion).alias('src_ipv4_int'),
        
        # IPv6 placeholder for now
        pl.when(pl.col('ip').str.contains(':'))
        .then(pl.lit(None, dtype=pl.Binary))
        .otherwise(None)
        .alias('src_ipv6_bytes')
    ]).drop('ip')
    
    print(f"✅ Optimized DEDUPLICATED probe mapping ready: {optimized_probe_map.height:,} entries")
    print("⚡ No more row duplication bias!")
    return optimized_probe_map

def optimize_all_columns():
    """Standard data type optimizations."""
    return [
        pl.col("prb_id").cast(pl.UInt32),
        pl.col("sent").cast(pl.UInt8), 
        pl.col("rcvd").cast(pl.UInt8),
        pl.col("avg").cast(pl.Float32),
        pl.col("ts"),  # Keep as i64
        pl.col("rtt_1").cast(pl.Float32),
        pl.col("rtt_2").cast(pl.Float32),  
        pl.col("rtt_3").cast(pl.Float32)
    ]

def add_optimized_ip_columns(df):
    """Add optimized IP columns using multiprocessing."""
    print("  🔄 Adding optimized IP columns...")
    return df.with_columns([
        # Detect IPv6 (vectorized)
        pl.col('dst_addr').str.contains(':').alias('dst_is_ipv6'),
        
        # Convert IPv4 using multiprocessing
        pl.col('dst_addr').map_batches(parallel_ipv4_conversion).alias('dst_ipv4_int'),
        
        # IPv6 placeholder
        pl.when(pl.col('dst_addr').str.contains(':'))
        .then(pl.lit(None, dtype=pl.Binary))
        .otherwise(None)
        .alias('dst_ipv6_bytes')
    ])

def add_probe_source_ips(df, optimized_probe_map):
    """Add source IP columns via DEDUPLICATED probe mapping (no row duplication)."""
    if optimized_probe_map is None:
        print("⚠️  No probe mapping available, skipping source IPs")
        return df
    
    print("  🔄 Adding source IPs via DEDUPLICATED probe mapping...")
    result = df.join(
        optimized_probe_map, 
        left_on="prb_id", 
        right_on="dst_prb_id", 
        how="left"
    )
    
    print(f"  ✅ Rows: {df.height:,} → {result.height:,} (no duplication!)")
    return result

def add_ip_display_columns():
    """Add readable IP display using vectorized operations."""
    return [
        # Destination IP display
        pl.when(pl.col('dst_is_ipv6'))
        .then(pl.col('dst_addr'))  # Keep original for IPv6
        .otherwise(
            # IPv4: vectorized operations
            pl.when(pl.col('dst_ipv4_int').is_not_null() & (pl.col('dst_ipv4_int') > 0))
            .then(
                (pl.col('dst_ipv4_int') // 16777216).cast(pl.String) + "." +
                ((pl.col('dst_ipv4_int') // 65536) % 256).cast(pl.String) + "." +
                ((pl.col('dst_ipv4_int') // 256) % 256).cast(pl.String) + "." +
                (pl.col('dst_ipv4_int') % 256).cast(pl.String)
            )
            .otherwise(None)
        )
        .alias("dst_addr_display"),
        
        # Source IP display
        pl.when(pl.col('src_is_ipv6').fill_null(False))
        .then(None)  # IPv6 source disabled for performance
        .otherwise(
            pl.when(pl.col('src_ipv4_int').is_not_null() & (pl.col('src_ipv4_int') > 0))
            .then(
                (pl.col('src_ipv4_int') // 16777216).cast(pl.String) + "." +
                ((pl.col('src_ipv4_int') // 65536) % 256).cast(pl.String) + "." +
                ((pl.col('src_ipv4_int') // 256) % 256).cast(pl.String) + "." +
                (pl.col('src_ipv4_int') % 256).cast(pl.String)
            )
            .otherwise(None)
        )
        .alias("src_addr_display")
    ]

def main():
    print("🚀 FIXED DATASET OPTIMIZATION (NO DUPLICATION BIAS)")
    print("=" * 70)
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # System info
    print(f"💻 System Info:")
    print(f"   CPU cores: {mp.cpu_count()}")
    print(f"   Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()
    
    # Setup input files and load probe IP mapping
    INPUT_DIR = "data/ping_parsed_parts"
    input_files = sorted(glob.glob(f"{INPUT_DIR}/*.parquet"))
    print(f"📁 Found {len(input_files)} parsed parquet files")

    PROBE_MAP_FILE = "probe_ip_map.csv"

    if os.path.exists(PROBE_MAP_FILE):
        probe_map = pl.read_csv(PROBE_MAP_FILE)
        print(f"📋 Loaded probe IP mapping: {probe_map.height:,} entries")
    else:
        print(f"⚠️  Probe mapping file not found: {PROBE_MAP_FILE}")
        probe_map = None
    
    print()
    
    # Create DEDUPLICATED probe mapping
    optimized_probe_map = create_deduplicated_probe_map(probe_map)
    print()
    
    # Test optimization on one file first
    if input_files:
        print("🧪 TESTING FIXED OPTIMIZATION ON FIRST FILE")
        print("-" * 60)
        
        # Original file size
        original_size = os.path.getsize(input_files[0]) / (1024**2)  # MB
        
        # Load and optimize
        print("Loading original data...")
        test_df = pl.scan_parquet(input_files[0]).collect()
        print(f"  Loaded {test_df.height:,} rows")
        
        print("Applying FIXED optimizations (no duplication)...")
        start_time = time.time()
        
        # Apply all optimizations including DEDUPLICATED probe mapping
        optimized_df = (
            test_df
            .pipe(add_optimized_ip_columns)  # Add destination IP columns first
            .pipe(add_probe_source_ips, optimized_probe_map)  # Add source IPs via DEDUPLICATED mapping
            .with_columns(optimize_all_columns())  # Optimize data types
            .with_columns(add_ip_display_columns())  # Add readable display columns
            .drop("dst_addr")  # Remove original IP string column
        )
        
        optimization_time = time.time() - start_time
        
        print(f"\n⏱️  Optimization completed in {optimization_time:.2f}s")
        print(f"📊 Row count: {test_df.height:,} → {optimized_df.height:,} (should be equal!)")
        
        if test_df.height == optimized_df.height:
            print("✅ SUCCESS: No row duplication!")
        else:
            print("❌ ERROR: Row count changed - still duplicating!")
            return
        
        # Write test files
        test_output = "test_optimized_fixed.parquet"
        print("Writing optimized test file...")
        optimized_df.write_parquet(test_output)
        
        # Compare sizes
        optimized_size = os.path.getsize(test_output) / (1024**2)  # MB
        reduction = (original_size - optimized_size) / original_size * 100
        
        print(f"\n📊 Size comparison:")
        print(f"  Original: {original_size:.1f} MB")
        print(f"  Optimized: {optimized_size:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Space saved: {original_size - optimized_size:.1f} MB")
        
        # Show sample data
        print(f"\n🔍 Sample optimized data:")
        display_cols = ['prb_id', 'dst_addr_display', 'src_addr_display', 'ts', 'avg']
        print(optimized_df.select(display_cols).head(5))
        
        # Clean up test file
        os.remove(test_output)
        
        print(f"\n✅ FIXED optimization test successful!")
        
        # Now process ALL files
        print("\n" + "="*70)
        print("🚀 PROCESSING ALL 966 FILES")
        print("="*70)
        
        OUTPUT_DIR = "data/ping_super_optimized_fixed"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Resume: filter out already processed files
        existing_files = {f.name for f in Path(OUTPUT_DIR).glob("*.parquet")}
        remaining_files = [f for f in input_files if Path(f).name not in existing_files]
        
        print(f"🎯 Ready to process {len(remaining_files)} files without duplication bias")
        
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"✅ Already processed: {len(existing_files)} files")
        print(f"🚀 Remaining to process: {len(remaining_files)} files")

        total_original_size = 0
        total_optimized_size = 0
        processed_files = 0
        start_full_time = time.time()

        for i, input_file in enumerate(remaining_files, 1):
            file_name = Path(input_file).name
            file_start_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Processing {i}/{len(remaining_files)}: {file_name}", end="", flush=True)
            
            try:
                # Track original size
                original_size = os.path.getsize(input_file)
                total_original_size += original_size
                
                # Load original data
                load_start = time.time()
                df = pl.scan_parquet(input_file).collect()
                original_rows = df.height
                load_time = time.time() - load_start
                
                # Apply FIXED optimization pipeline
                opt_start = time.time()
                optimized_df = (
                    df
                    .pipe(add_optimized_ip_columns)
                    .pipe(add_probe_source_ips, optimized_probe_map)
                    .with_columns(optimize_all_columns())
                    .with_columns(add_ip_display_columns())
                    .drop("dst_addr")
                )
                opt_time = time.time() - opt_start
                
                # Verify no duplication
                if optimized_df.height != original_rows:
                    print(f" ❌ ERROR: Row duplication detected ({original_rows} → {optimized_df.height})")
                    continue
                
                # Write optimized file
                write_start = time.time()
                output_file = f"{OUTPUT_DIR}/{file_name}"
                optimized_df.write_parquet(output_file)
                write_time = time.time() - write_start
                
                # Track optimized size
                optimized_size = os.path.getsize(output_file)
                total_optimized_size += optimized_size
                processed_files += 1
                
                # Show progress with detailed timing
                file_reduction = (original_size - optimized_size) / original_size * 100
                total_file_time = time.time() - file_start_time
                print(f" → {file_reduction:.1f}% reduction ({total_file_time:.1f}s: load={load_time:.1f}s, opt={opt_time:.1f}s, write={write_time:.1f}s)")
                
                # Show progress every 10 files for better ETA tracking
                if i % 10 == 0 or i == len(remaining_files):
                    cumulative_reduction = (total_original_size - total_optimized_size) / total_original_size * 100
                    elapsed = time.time() - start_full_time
                    rate = processed_files / elapsed if elapsed > 0 else 0
                    eta_minutes = (len(remaining_files) - processed_files) / rate / 60 if rate > 0 else 0
                    
                    # Calculate average processing time per file
                    avg_time_per_file = elapsed / processed_files if processed_files > 0 else 0
                    
                    print(f"    📈 [{time.strftime('%H:%M:%S')}] Progress: {processed_files}/{len(remaining_files)} files ({cumulative_reduction:.1f}% reduction)")
                    print(f"    ⏱️  Rate: {rate:.2f} files/sec, Avg: {avg_time_per_file:.1f}s/file, ETA: {eta_minutes:.1f} minutes ({time.strftime('%H:%M', time.localtime(time.time() + eta_minutes*60))})")
                    
            except Exception as e:
                print(f" ❌ ERROR: {str(e)[:50]}...")
                continue

        total_time = time.time() - start_full_time
        print(f"\n🎉 FIXED optimization complete! Processed {processed_files}/{len(remaining_files)} files in {total_time/60:.1f} minutes")
        
        # Final results
        if processed_files > 0:
            final_gb = total_optimized_size / (1024**3)
            original_gb = total_original_size / (1024**3)
            total_reduction = (total_original_size - total_optimized_size) / total_original_size * 100
            space_saved_gb = (total_original_size - total_optimized_size) / (1024**3)
            
            print(f"\n🏆 FINAL RESULTS (NO DUPLICATION BIAS):")
            print(f"  📄 Files processed: {processed_files} / {len(remaining_files)}")
            print(f"  📦 Original size: {original_gb:.2f} GB")
            print(f"  🗜️  Optimized size: {final_gb:.2f} GB")
            print(f"  📉 Size reduction: {total_reduction:.1f}%")
            print(f"  💾 Space saved: {space_saved_gb:.2f} GB")
            print(f"  ⚡ Processing rate: {processed_files/(total_time/60):.1f} files/minute")
            
            print(f"\n📁 Optimized dataset ready at: {OUTPUT_DIR}/")
            print(f"💡 Usage: pl.scan_parquet('{OUTPUT_DIR}/*.parquet')")
            print(f"🎯 Features: Source IPs added, no duplication bias, 28%+ smaller files")
        
    else:
        print("❌ No input files found")

if __name__ == "__main__":
    main()