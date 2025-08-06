# IPv4 Filtering Analysis Summary

## Overview
Analysis of ping data in `data/ping_super_optimized_fixed/` to determine the impact of filtering to only include measurements where BOTH source and destination have IPv4 addresses (non-null `src_ipv4_int` AND non-null `dst_ipv4_int`).

## Dataset Scale
- **Total files analyzed**: 964 parquet files
- **Processing time**: 8.9 minutes (532 seconds)

## Main Results

### Original Dataset
- **Total rows**: 27,228,978,138 (27.23 billion rows)
- **Size**: 27.23 billion measurements

### After IPv4 Filtering
- **Rows remaining**: 26,166,781,745 (26.17 billion rows)
- **Percentage retained**: 96.10%
- **Rows removed**: 1,062,196,393 (1.06 billion rows)
- **Percentage removed**: 3.90%

## Impact Analysis

### Storage and Performance
- **Storage savings**: ~3.9%
- **Processing speedup**: ~1.04x faster
- **Reduction factor**: 1.04x smaller dataset

### What Gets Filtered Out
Based on detailed analysis of sample files:

1. **IPv6 Source Addresses**: 3.08% of rows have null `src_ipv4_int` (IPv6 sources)
2. **IPv6 Destination Addresses**: Small percentage have IPv6 destinations
3. **Mixed Protocol Pairs**: 
   - IPv4 source → IPv6 destination: 41.33% of sample
   - IPv6 source → IPv4 destination: 0.23% of sample

### Address Type Distribution
- **Both IPv4 (src & dst)**: 96.92% (these would remain)
- **Both IPv6 (src & dst)**: 0.17% (these would be removed)
- **Mixed combinations**: Various percentages would be removed

## Examples of Filtered Data

### IPv6 Destinations (would be removed):
```
prb_id: 6878, src: 31.130.200.18 (IPv4) → dst: 2001:4ba0:ffe0:ffff::4 (IPv6)
prb_id: 6878, src: 31.130.200.18 (IPv4) → dst: 2a01:9e01:4d05:3333::a (IPv6)
```

### IPv6 Sources (would be removed):
```
prb_id: 1004626, src: null (IPv6) → dst: 45.68.49.82 (IPv4)
prb_id: 50016, src: null (IPv6) → dst: 2001:500:1::53 (IPv6)
```

## Recommendation

The filtering would retain **96.10%** of the data, removing only **3.90%** of measurements. This represents a minimal data loss while significantly simplifying analysis by ensuring all remaining measurements have both IPv4 source and destination addresses available as integers for efficient processing.

### Benefits of filtering:
1. **Consistent data types**: All remaining rows have IPv4 addresses as integers
2. **Simplified analysis**: No need to handle mixed IPv4/IPv6 cases
3. **Performance gains**: 4% smaller dataset, slightly faster processing
4. **Memory efficiency**: Reduced memory usage by ~4%

### Trade-offs:
1. **Loss of IPv6 data**: Complete removal of IPv6-related measurements
2. **Reduced coverage**: Some network paths involving IPv6 will be missing
3. **Potential bias**: May underrepresent modern IPv6 adoption in analysis

The filtering appears to be reasonable for IPv4-focused analysis, with minimal impact on overall dataset size while ensuring data consistency.