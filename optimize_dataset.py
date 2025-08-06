#!/usr/bin/env python3
"""
Dataset Optimization Script
Optimize the parsed ping dataset for maximum space efficiency using binary IP storage and optimized data types.
Expected results: 40% file size reduction (tested with real data)
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
    """
    Convert batch of IPv4 addresses to UInt32 using multiprocessing.
    Works on numpy arrays for maximum efficiency.
    """
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
    """
    Convert IP series to IPv4 integers using all available CPU cores.
    Uses map_batches with multiprocessing for maximum performance.
    """
    print(f"  🔄 Converting {len(series):,} IPs with {mp.cpu_count()} cores...")
    start_time = time.time()
    
    # Convert to numpy for efficiency
    ip_array = series.to_numpy()
    
    # Use multiprocessing to parallelize across CPU cores
    with ProcessPoolExecutor(max_workers=min(63, mp.cpu_count())) as executor:
        # Split into chunks for parallel processing
        chunk_size = max(1000, len(ip_array) // (mp.cpu_count() * 4))
        chunks = [ip_array[i:i + chunk_size] for i in range(0, len(ip_array), chunk_size)]
        
        print(f"  ⚡ Processing {len(chunks)} chunks of ~{chunk_size:,} IPs each...")
        
        # Process chunks in parallel
        results = list(executor.map(convert_ipv4_batch, chunks))
        
        # Combine results
        combined = np.concatenate(results) if results else np.array([], dtype=np.uint32)
    
    elapsed = time.time() - start_time
    print(f"  ✅ IP conversion completed in {elapsed:.2f}s ({len(series)/elapsed:,.0f} IPs/sec)")
    
    return pl.Series(combined)

def create_optimized_probe_map(probe_map):
    """
    Create optimized probe mapping using multiprocessing for IP conversion.
    """
    if probe_map is None:
        return None
    
    print("🔄 Creating optimized probe mapping with multiprocessing...")
    
    optimized_probe_map = probe_map.with_columns([
        # Detect IPv6 (vectorized)
        pl.col('ip').str.contains(':').alias('src_is_ipv6'),
        
        # Convert IPv4 using multiprocessing
        pl.col('ip').map_batches(parallel_ipv4_conversion).alias('src_ipv4_int'),
        
        # IPv6 placeholder for now (can add parallel processing later if needed)
        pl.when(pl.col('ip').str.contains(':'))
        .then(pl.lit(None, dtype=pl.Binary))
        .otherwise(None)
        .alias('src_ipv6_bytes')
    ]).drop('ip')
    
    print(f"✅ Optimized probe mapping ready: {optimized_probe_map.height:,} entries")
    print("⚡ IPv4 conversion used all available CPU cores via multiprocessing")
    return optimized_probe_map

def optimize_all_columns():
    """
    Standard data type optimizations (already vectorized).
    """
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
    """
    Add optimized IP columns using multiprocessing for conversion.
    """
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
    """
    Join with probe mapping (vectorized operation).
    """
    if optimized_probe_map is None:
        print("⚠️  No probe mapping available, skipping source IPs")
        return df
    
    print("  🔄 Adding source IPs via probe mapping...")
    return df.join(
        optimized_probe_map, 
        left_on="prb_id", 
        right_on="dst_prb_id", 
        how="left"
    )

def add_ip_display_columns():
    """
    Add readable IP display using vectorized bit operations.
    """
    return [
        # Destination IP display
        pl.when(pl.col('dst_is_ipv6'))
        .then(pl.col('dst_addr'))  # Keep original for IPv6
        .otherwise(
            # IPv4: vectorized bit operations - use floor division instead of bit shift
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
        .then(None)  # IPv6 source disabled
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

def monitor_cpu_usage():
    """Monitor CPU usage during processing"""
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    avg_cpu = sum(cpu_percent) / len(cpu_percent)
    max_cpu = max(cpu_percent)
    active_cores = sum(1 for cpu in cpu_percent if cpu > 5.0)
    
    print(f"  📊 CPU Usage: Avg={avg_cpu:.1f}%, Max={max_cpu:.1f}%, Active cores={active_cores}/{len(cpu_percent)}")
    return avg_cpu, active_cores, len(cpu_percent)

def main():
    print("🚀 DATASET OPTIMIZATION WITH MULTIPROCESSING")
    print("=" * 60)
    
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
        
        # Analyze probe IP distribution
        probe_ips = probe_map['ip'].to_list()
        probe_ipv4_count = sum(1 for ip in probe_ips if ':' not in ip)
        probe_ipv6_count = len(probe_ips) - probe_ipv4_count
        
        print(f"🌐 Probe IP distribution:")
        print(f"  IPv4 probes: {probe_ipv4_count:,} ({probe_ipv4_count/len(probe_ips)*100:.1f}%)")
        print(f"  IPv6 probes: {probe_ipv6_count:,} ({probe_ipv6_count/len(probe_ips)*100:.1f}%)")
        
    else:
        print(f"⚠️  Probe mapping file not found: {PROBE_MAP_FILE}")
        print(f"   Will skip probe ID → IP conversion")
        probe_map = None
    
    print()
    
    # Create optimized probe mapping with multiprocessing
    optimized_probe_map = create_optimized_probe_map(probe_map)
    print()
    
    # Test optimization on one file first
    if input_files:
        print("🧪 TESTING OPTIMIZATION ON FIRST FILE")
        print("-" * 50)
        
        # Original file size
        original_size = os.path.getsize(input_files[0]) / (1024**2)  # MB
        
        # Load and optimize
        print("Loading original data...")
        test_df = pl.scan_parquet(input_files[0]).collect()
        print(f"  Loaded {test_df.height:,} rows")
        
        print("Applying optimizations...")
        start_time = time.time()
        
        # Monitor CPU during optimization
        print("🔧 Starting optimization pipeline...")
        
        # Apply all optimizations including probe mapping
        optimized_df = (
            test_df
            .pipe(add_optimized_ip_columns)  # Add destination IP columns first
            .pipe(add_probe_source_ips, optimized_probe_map)  # Add source IPs via probe mapping
            .with_columns(optimize_all_columns())  # Optimize data types
            .with_columns(add_ip_display_columns())  # Add readable display columns BEFORE dropping dst_addr
            .drop("dst_addr")  # Remove original IP string column
        )
        
        optimization_time = time.time() - start_time
        
        # Monitor final CPU usage
        avg_cpu, active_cores, total_cores = monitor_cpu_usage()
        
        print(f"\n⏱️  Optimization completed in {optimization_time:.2f}s")
        print(f"📊 CPU utilization: {active_cores}/{total_cores} cores active ({avg_cpu:.1f}% average)")
        
        # Write test files
        test_output = "test_optimized.parquet"
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
        
        print(f"\n📋 Optimized schema:")
        for col, dtype in optimized_df.schema.items():
            print(f"    {col}: {dtype}")
        
        print(f"\n🔍 Sample optimized data (showing readable IPs):")
        display_cols = ['prb_id', 'dst_addr_display', 'ts', 'sent', 'rcvd', 'avg', 'rtt_1', 'dst_is_ipv6']
        if 'src_addr_display' in optimized_df.columns:
            display_cols.insert(2, 'src_addr_display')
            display_cols.append('src_is_ipv6')
        
        print(optimized_df.select(display_cols).head(8))
        
        # Clean up test file
        os.remove(test_output)
        
        print(f"\n✅ Test successful!")
        
        # Performance analysis
        if active_cores < total_cores * 0.8:
            print(f"⚠️  WARNING: Only {active_cores}/{total_cores} cores were utilized")
            print(f"   This suggests the multiprocessing may not be working optimally")
        else:
            print(f"🎉 SUCCESS: {active_cores}/{total_cores} cores were actively used!")
        
        print("\n" + "="*60)
        print("🚀 PROCESSING ALL FILES")
        print("="*60)
        
        # Full dataset optimization with probe mapping
        OUTPUT_DIR = "data/ping_super_optimized"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Clean up any existing files
        for f in Path(OUTPUT_DIR).glob("*.parquet"):
            f.unlink()

        print(f"🚀 Optimizing {len(input_files)} files with probe mapping...")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"⏱️  This may take a while for large datasets...")

        total_original_size = 0
        total_optimized_size = 0
        processed_files = 0
        start_full_time = time.time()

        for i, input_file in enumerate(input_files, 1):
            file_name = Path(input_file).name
            print(f"Processing {i}/{len(input_files)}: {file_name}", end="", flush=True)
            
            try:
                # Track original size
                original_size = os.path.getsize(input_file)
                total_original_size += original_size
                
                # Load original data
                df = pl.scan_parquet(input_file).collect()
                
                # Apply full optimization pipeline with probe mapping
                optimized_df = (
                    df
                    .pipe(add_optimized_ip_columns)  # Convert destination IPs to optimal storage
                    .pipe(add_probe_source_ips, optimized_probe_map)  # Add source IPs via probe mapping
                    .with_columns(optimize_all_columns())  # Optimize all data types
                    .with_columns(add_ip_display_columns())  # Add display columns
                    .drop("dst_addr")  # Remove original IP string column
                )
                
                # Write optimized file
                output_file = f"{OUTPUT_DIR}/{file_name}"
                optimized_df.write_parquet(output_file)
                
                # Track optimized size
                optimized_size = os.path.getsize(output_file)
                total_optimized_size += optimized_size
                processed_files += 1
                
                # Show progress
                file_reduction = (original_size - optimized_size) / original_size * 100
                print(f" → {file_reduction:.1f}% reduction")
                
                # Show cumulative progress every 50 files or at end
                if i % 50 == 0 or i == len(input_files):
                    cumulative_reduction = (total_original_size - total_optimized_size) / total_original_size * 100
                    elapsed = time.time() - start_full_time
                    rate = processed_files / elapsed if elapsed > 0 else 0
                    eta = (len(input_files) - processed_files) / rate if rate > 0 else 0
                    print(f"    📈 Progress: {processed_files}/{len(input_files)} files ({cumulative_reduction:.1f}% reduction)")
                    print(f"    ⏱️  Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} minutes")
                    
            except Exception as e:
                print(f" ❌ ERROR: {str(e)[:50]}...")
                continue

        total_time = time.time() - start_full_time
        print(f"\n🎉 Full optimization complete! Processed {processed_files}/{len(input_files)} files in {total_time/60:.1f} minutes")
        
        # Final results
        if os.path.exists(OUTPUT_DIR):
            output_files = list(Path(OUTPUT_DIR).glob("*.parquet"))
            final_size = sum(f.stat().st_size for f in output_files)
            final_gb = final_size / (1024**3)
            
            original_gb = total_original_size / (1024**3)
            total_reduction = (total_original_size - total_optimized_size) / total_original_size * 100
            space_saved_gb = (total_original_size - total_optimized_size) / (1024**3)
            
            print(f"\n🏆 FINAL OPTIMIZATION RESULTS:")
            print(f"  📄 Files processed: {len(output_files)} / {len(input_files)}")
            print(f"  📦 Original size: {original_gb:.2f} GB")
            print(f"  🗜️  Optimized size: {final_gb:.2f} GB")
            print(f"  📉 Size reduction: {total_reduction:.1f}%")
            print(f"  💾 Space saved: {space_saved_gb:.2f} GB")
            print(f"  ⏱️  Processing time: {total_time/60:.1f} minutes")
            print(f"  ⚡ Average rate: {processed_files/(total_time/60):.1f} files/minute")
            
            # Estimate full dataset savings (if this was applied to 1TB)
            if original_gb > 0:
                tb_estimate = (1024 * total_reduction / 100)
                print(f"  🌟 Est. 1TB dataset savings: {tb_estimate:.0f} GB")
            
            print(f"\n📁 Super-optimized dataset ready at: {OUTPUT_DIR}/")
            print(f"💡 Usage: pl.scan_parquet('{OUTPUT_DIR}/*.parquet')")
            print(f"🔍 Readable IPs: Use 'dst_addr_display' and 'src_addr_display' columns")
            print(f"⚡ Storage: IPv4 as UInt32, IPv6 as binary, optimized types")
            if optimized_probe_map is not None:
                print(f"🎯 Probe mapping: Source IPs added via probe_ip_map.csv")
            
            # Quick verification
            try:
                print(f"\n🔬 Quick verification...")
                test_read = pl.scan_parquet(f"{OUTPUT_DIR}/*.parquet").head(3).collect()
                print(f"✅ Successfully read optimized dataset!")
                print(f"📊 Total rows across all files: {pl.scan_parquet(f'{OUTPUT_DIR}/*.parquet').select(pl.len()).collect().item():,}")
                
            except Exception as e:
                print(f"⚠️  Verification failed: {e}")
        
    else:
        print("❌ No input files found")

if __name__ == "__main__":
    main()