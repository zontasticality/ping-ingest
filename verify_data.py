#!/usr/bin/env python3
"""
Data Integrity Verification Script
Compare original and optimized datasets to ensure data integrity
"""

import polars as pl

def main():
    print('🔍 DATA INTEGRITY CHECK')
    print('=' * 50)

    # Read first 100,000 rows from original data
    print('Loading original data (first 100k rows)...')
    original = pl.scan_parquet('data/ping_parsed_parts/part_0001.parquet').head(100000).collect()
    print(f'Original: {original.height:,} rows, {len(original.columns)} columns')

    # Read first 100,000 rows from optimized data  
    print('Loading optimized data (first 100k rows)...')
    optimized = pl.scan_parquet('data/ping_super_optimized/part_0001.parquet').head(100000).collect()
    print(f'Optimized: {optimized.height:,} rows, {len(optimized.columns)} columns')

    print('\n📋 SCHEMA COMPARISON:')
    print('Original columns:', original.columns)
    print('Optimized columns:', optimized.columns)

    print('\n🔍 DATA COMPARISON:')

    # Compare core data fields (excluding new optimized columns)
    core_fields = ['prb_id', 'ts', 'sent', 'rcvd', 'avg', 'rtt_1', 'rtt_2', 'rtt_3', 'dst_addr']

    for field in core_fields:
        if field in original.columns:
            if field == 'dst_addr':
                # Compare destination addresses
                orig_ips = original[field].to_list()
                opt_display_ips = optimized['dst_addr_display'].to_list()
                
                mismatches = 0
                for i in range(min(len(orig_ips), len(opt_display_ips))):
                    if orig_ips[i] != opt_display_ips[i]:
                        mismatches += 1
                        if mismatches <= 5:  # Show first 5 mismatches
                            print(f'  ❌ Row {i}: {orig_ips[i]} != {opt_display_ips[i]}')
                
                if mismatches == 0:
                    print(f'  ✅ {field} → dst_addr_display: ALL {len(orig_ips):,} IPs match perfectly!')
                else:
                    print(f'  ⚠️  {field} → dst_addr_display: {mismatches:,} mismatches out of {len(orig_ips):,}')
            else:
                # Compare numeric fields
                orig_vals = original[field].to_list()
                opt_vals = optimized[field].to_list()
                
                mismatches = 0
                for i in range(min(len(orig_vals), len(opt_vals))):
                    if orig_vals[i] != opt_vals[i]:
                        mismatches += 1
                        if mismatches <= 3:  # Show first 3 mismatches
                            print(f'  ❌ Row {i}: {orig_vals[i]} != {opt_vals[i]}')
                
                if mismatches == 0:
                    print(f'  ✅ {field}: ALL {len(orig_vals):,} values match perfectly!')
                else:
                    print(f'  ⚠️  {field}: {mismatches:,} mismatches out of {len(orig_vals):,}')

    # Check if source IPs were added correctly
    print('\n🎯 SOURCE IP VERIFICATION:')
    src_ips_added = optimized['src_addr_display'].drop_nulls().len()
    total_rows = optimized.height
    print(f'  Source IPs added: {src_ips_added:,} out of {total_rows:,} rows ({src_ips_added/total_rows*100:.1f}%)')

    # Show sample of data side by side
    print('\n📊 SAMPLE DATA COMPARISON (first 5 rows):')
    print('Original:')
    print(original[['prb_id', 'dst_addr', 'ts', 'avg', 'rtt_1']].head(5))
    print('\nOptimized:')
    print(optimized[['prb_id', 'dst_addr_display', 'src_addr_display', 'ts', 'avg', 'rtt_1']].head(5))

    print('\n🔬 DATA TYPES OPTIMIZATION:')
    for col in ['prb_id', 'sent', 'rcvd', 'avg', 'rtt_1']:
        if col in optimized.columns:
            orig_type = original.schema[col] if col in original.schema else 'N/A'
            opt_type = optimized.schema[col]
            print(f'  {col}: {orig_type} → {opt_type}')

    # Additional verification: Check binary IP storage
    print('\n🔧 IP STORAGE VERIFICATION:')
    ipv4_count = optimized.filter(~pl.col('dst_is_ipv6')).height
    ipv6_count = optimized.filter(pl.col('dst_is_ipv6')).height
    print(f'  IPv4 destinations: {ipv4_count:,} (stored as UInt32)')
    print(f'  IPv6 destinations: {ipv6_count:,} (stored as Binary)')
    
    # Verify a few IPv4 conversions manually
    print('\n🧮 MANUAL IP CONVERSION VERIFICATION:')
    sample_rows = optimized.head(10).select(['dst_addr_display', 'dst_ipv4_int', 'dst_is_ipv6'])
    for i, row in enumerate(sample_rows.iter_rows()):
        display_ip, int_ip, is_ipv6 = row
        if not is_ipv6 and int_ip is not None and display_ip is not None:
            # Manually convert integer back to IP
            manual_ip = f"{(int_ip >> 24) & 255}.{(int_ip >> 16) & 255}.{(int_ip >> 8) & 255}.{int_ip & 255}"
            match = manual_ip == display_ip
            status = '✅' if match else '❌'
            print(f'  {status} Row {i}: {display_ip} ↔ UInt32({int_ip}) ↔ {manual_ip}')
            if not match:
                break
    
    print('\n✅ DATA INTEGRITY CHECK COMPLETE')

if __name__ == "__main__":
    main()