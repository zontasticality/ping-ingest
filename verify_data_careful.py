#!/usr/bin/env python3
"""
Careful Data Integrity Verification Script
"""

import polars as pl

def main():
    print('🔍 CAREFUL DATA INTEGRITY CHECK')
    print('=' * 50)

    # Read smaller sample first to be more careful
    print('Loading original data (first 1000 rows)...')
    original = pl.scan_parquet('data/ping_parsed_parts/part_0001.parquet').head(1000).collect()
    print(f'Original: {original.height:,} rows, {len(original.columns)} columns')

    print('Loading optimized data (first 1000 rows)...')
    optimized = pl.scan_parquet('data/ping_super_optimized/part_0001.parquet').head(1000).collect()
    print(f'Optimized: {optimized.height:,} rows, {len(optimized.columns)} columns')

    print('\n📋 DETAILED FIRST 10 ROWS COMPARISON:')
    
    # Show first 10 rows side by side for manual inspection
    print('Original (first 10 rows):')
    orig_sample = original[['prb_id', 'dst_addr', 'ts', 'sent', 'rcvd', 'avg']].head(10)
    print(orig_sample)
    
    print('\nOptimized (first 10 rows):')
    opt_sample = optimized[['prb_id', 'dst_addr_display', 'src_addr_display', 'ts', 'sent', 'rcvd', 'avg']].head(10)
    print(opt_sample)
    
    print('\n🔍 ROW-BY-ROW VERIFICATION (first 10 rows):')
    
    for i in range(10):
        orig_row = original.row(i)
        opt_row = optimized.row(i)
        
        print(f'\nRow {i}:')
        print(f'  Original: prb_id={orig_row[0]}, dst_addr={orig_row[1]}, ts={orig_row[2]}, avg={orig_row[5]}')
        print(f'  Optimized: prb_id={opt_row[0]}, dst_addr_display={opt_row[14]}, ts={opt_row[1]}, avg={opt_row[4]}')
        
        # Check matches
        matches = []
        matches.append(f"prb_id: {orig_row[0] == opt_row[0]}")
        matches.append(f"dst_addr: {orig_row[1] == opt_row[14]}")  # dst_addr vs dst_addr_display
        matches.append(f"ts: {orig_row[2] == opt_row[1]}")
        matches.append(f"avg: {orig_row[5] == opt_row[4]}")
        
        print(f'  Matches: {", ".join(matches)}')
    
    print('\n🔧 COLUMN MAPPING CHECK:')
    print('Original schema:')
    for i, (col, dtype) in enumerate(original.schema.items()):
        print(f'  [{i}] {col}: {dtype}')
    
    print('\nOptimized schema:')
    for i, (col, dtype) in enumerate(optimized.schema.items()):
        print(f'  [{i}] {col}: {dtype}')
    
    print('\n🎯 STATISTICAL COMPARISON:')
    
    # Compare some basic statistics
    print('Probe ID statistics:')
    print(f'  Original min/max: {original["prb_id"].min()} / {original["prb_id"].max()}')
    print(f'  Optimized min/max: {optimized["prb_id"].min()} / {optimized["prb_id"].max()}')
    
    print('Timestamp statistics:')
    print(f'  Original min/max: {original["ts"].min()} / {original["ts"].max()}')
    print(f'  Optimized min/max: {optimized["ts"].min()} / {optimized["ts"].max()}')
    
    print('Unique IP count:')
    orig_unique_ips = original["dst_addr"].n_unique()
    opt_unique_ips = optimized["dst_addr_display"].n_unique()
    print(f'  Original unique IPs: {orig_unique_ips}')
    print(f'  Optimized unique IPs: {opt_unique_ips}')
    
    # Check if data might be reordered
    print('\n🔄 ORDER VERIFICATION:')
    print('First 5 probe IDs in original:', original["prb_id"].head(5).to_list())
    print('First 5 probe IDs in optimized:', optimized["prb_id"].head(5).to_list())
    
    print('First 5 timestamps in original:', original["ts"].head(5).to_list())
    print('First 5 timestamps in optimized:', optimized["ts"].head(5).to_list())
    
    print('\n✅ CAREFUL VERIFICATION COMPLETE')

if __name__ == "__main__":
    main()