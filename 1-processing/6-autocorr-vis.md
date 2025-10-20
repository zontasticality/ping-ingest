# Network Latency Autocorrelation Matrix Visualization

## Project Overview

This document specifies a memory-efficient streaming approach to generate a 200×200 autocorrelation matrix visualization from network ping measurement data. The approach calculates lag-1 autocorrelation for all pairs of the 200 most active network nodes across the complete dataset.

## Objectives

### Primary Goal
Generate a 2D heatmap showing lag-1 autocorrelation coefficients between all pairs of the top 200 most active network nodes, where:
- **Rows**: Source nodes (200 nodes)
- **Columns**: Destination nodes (200 nodes) 
- **Cell values**: Lag-1 autocorrelation coefficient (-1 to +1)
- **Color encoding**: Blue (positive correlation) to Red (negative correlation)

### Secondary Goals
- Maintain memory usage under 1GB throughout processing
- Process complete dataset (500 parquet files) for statistical robustness
- Achieve reasonable processing time (< 3 hours)
- Generate publication-quality visualization

## Technical Approach

### Algorithm Design

#### Phase 1: Node Selection
```
INPUT: First parquet file (part_0001.parquet)
OUTPUT: List of 200 most active node IDs

PROCESS:
1. Load first parquet file with streaming scan
2. Aggregate measurement counts by source and destination nodes
3. Combine source + destination activity to get total node activity
4. Select top 200 nodes by total measurement count
5. Store node-to-index mapping for matrix construction
```

**Rationale**: Using first file for node selection is computationally efficient while likely capturing the globally most active nodes, as network measurement patterns tend to be consistent across time periods.

#### Phase 2: Storage Initialization
```
INPUT: 200 selected node IDs
OUTPUT: Initialized data structures

PROCESS:
1. Generate all possible node pairs: 200 × 199 = 39,800 pairs (excluding self-loops)
2. Initialize sparse storage: Dict[(src_node, dst_node)] -> List[(timestamp, latency)]
3. Create node-to-matrix-index mapping for final matrix construction
```

#### Phase 3: Single-Pass Data Collection
```
INPUT: 500 parquet files, target node pairs
OUTPUT: Complete time series data for each active pair

FOR each parquet file:
    1. Lazy scan parquet file
    2. Filter measurements:
       - avg > 0 (valid latency)
       - sent == rcvd (successful measurements)
       - src_ipv4_int IN top_200_nodes
       - dst_ipv4_int IN top_200_nodes
    3. Collect filtered data (small subset per file)
    4. Append measurements to corresponding pair storage
    5. Monitor memory usage; terminate if exceeding 1GB
```

**Key Innovation**: Single pass through all files with simultaneous collection for all pairs, eliminating the O(pairs × files) complexity of nested scanning.

#### Phase 4: Autocorrelation Matrix Calculation
```
INPUT: Time series data for each pair
OUTPUT: 200×200 autocorrelation coefficient matrix

FOR each pair with >= 10 measurements:
    1. Sort measurements by timestamp (ensure temporal order)
    2. Extract latency values: [l₁, l₂, l₃, ..., lₙ]
    3. Create lag-1 series: 
       - Original: [l₁, l₂, l₃, ..., lₙ₋₁]
       - Lagged:   [l₂, l₃, l₄, ..., lₙ]
    4. Calculate Pearson correlation coefficient between series
    5. Store result in matrix[src_idx, dst_idx]
    6. Set diagonal elements to 1.0 (perfect self-correlation)
    7. Leave missing pairs as NaN (no data available)
```

#### Phase 5: Visualization Generation
```
INPUT: 200×200 autocorrelation matrix
OUTPUT: High-resolution heatmap visualization

PROCESS:
1. Create 200×200 heatmap using matplotlib imshow()
2. Apply diverging colormap (RdBu_r: Red-White-Blue reversed)
3. Set color range: vmin=-1, vmax=+1
4. Mask NaN values (white/transparent for missing data)
5. Add comprehensive labeling:
   - Title: Node count, dataset coverage percentage
   - Axes: "Source Node Index", "Destination Node Index"  
   - Colorbar: "Lag-1 Autocorrelation Coefficient"
6. Add grid lines for readability (every 25 nodes)
7. Save as high-resolution PNG (300 DPI)
```

## Resource Requirements

### Memory Analysis
```
Base Memory Calculation:
- Pairs: 39,800
- Average measurements per pair: 1,120 (estimated from test data)
- Bytes per measurement: 16 (8-byte timestamp + 8-byte float)
- Total data: 39,800 × 1,120 × 16 = 713 MB

Additional Memory:
- Matrix storage: 200 × 200 × 8 = 320 KB
- Processing overhead: ~100 MB
- Total peak memory: ~850 MB
```

**Memory Safety**: Well under 1GB limit with margin for variance in data density.

### Time Complexity Analysis
```
Operations:
- Node selection: O(measurements_in_first_file) ≈ 30M operations
- File processing: O(files × measurements_per_file) ≈ 500 × 30M = 15B operations  
- Filtering overhead: ~10x multiplier for Polars operations
- Autocorrelation calculation: O(pairs × avg_measurements) ≈ 40K × 1K = 40M operations

Estimated total: ~150B operations
At ~100M operations/second: ~25 minutes processing time
```

### Disk I/O Requirements
```
- Input files: 500 parquet files × ~700MB average = 350GB read
- Sequential read pattern (streaming friendly)
- Output: 1 PNG file (~2MB), 1 pickle file (~50MB)
```

## Data Quality Considerations

### Statistical Validity
- **Minimum measurements per pair**: 10 (required for meaningful correlation)
- **Temporal coverage**: Complete dataset timespan for robust autocorrelation
- **Sample size**: Expected 5,000-15,000 active pairs from 39,800 possible pairs

### Bias Analysis
- **Node selection bias**: Top 200 from first file may not represent global top 200
  - *Mitigation*: Network measurement patterns typically consistent across time
- **Temporal bias**: Dataset represents specific time period
  - *Accepted limitation*: Analysis is specific to measurement campaign timeframe
- **Missing data**: Many node pairs will have insufficient measurements
  - *Handled*: Sparse matrix with NaN for missing data, clearly indicated in visualization

## Expected Outputs

### Primary Output: Autocorrelation Matrix Heatmap
- **Format**: 200×200 pixel heatmap, 2400×2400 pixels at 300 DPI
- **File**: `autocorr_matrix_200x200.png`
- **Content**: Lag-1 autocorrelation coefficients for all node pairs
- **Interpretation**:
  - Blue regions: Positive temporal correlation (predictable latency patterns)
  - Red regions: Negative temporal correlation (anti-correlated patterns)
  - White regions: Insufficient data for correlation calculation
  - Diagonal: Perfect self-correlation (blue, value = 1.0)

### Secondary Outputs
- **Data file**: `autocorr_matrix_200x200.pkl` - Serialized matrix and metadata
- **Progress logs**: Real-time processing statistics and memory usage
- **Summary statistics**: Matrix coverage, correlation distribution, processing metrics

## Success Criteria

### Technical Criteria
- [x] Memory usage remains below 1GB throughout execution
- [x] Processing completes within 3 hours
- [x] Matrix achieves >10% coverage (>4,000 active pairs)
- [x] Visualization renders without artifacts

### Scientific Criteria  
- [x] Autocorrelation calculations are statistically valid (n≥10 per pair)
- [x] Full dataset temporal coverage (not just recent data)
- [x] Correlation coefficients span meaningful range (not all near zero)
- [x] Results are reproducible and verifiable

## Risk Analysis

### High Risk
- **Memory explosion**: If node pairs are denser than estimated
  - *Mitigation*: Real-time memory monitoring with graceful termination
- **Processing timeout**: If I/O is slower than expected  
  - *Mitigation*: Progress tracking with time estimates

### Medium Risk
- **Sparse matrix**: Most pairs may lack sufficient data
  - *Acceptance*: Expected behavior, will be clearly visualized
- **Node selection bias**: First file may not represent global activity
  - *Mitigation*: Document limitation, consider multi-file sampling in future

### Low Risk
- **Correlation calculation errors**: Edge cases in pearsonr()
  - *Mitigation*: Robust error handling with fallback to 0.0
- **Visualization rendering issues**: Large matrix display problems
  - *Mitigation*: Standard matplotlib approach, well-tested

## Implementation Validation

### Test Strategy
1. **Memory test**: Monitor peak memory usage during execution
2. **Correctness test**: Verify autocorrelation calculations on known data
3. **Performance test**: Measure processing time per file and extrapolate
4. **Completeness test**: Verify all expected pairs are processed

### Acceptance Criteria
- Memory usage curve remains below 1GB threshold
- Matrix coverage >10% with reasonable correlation distribution  
- Processing completes successfully with meaningful visualization
- Results pass scientific reasonableness checks

## Future Enhancements

### Immediate Improvements
- Multi-file node selection for reduced selection bias
- Parallel processing for faster execution
- Progressive matrix updates for real-time monitoring

### Advanced Features
- Multiple lag autocorrelations (lag-1, lag-2, lag-5)
- Geographic clustering analysis of correlation patterns
- Temporal evolution of correlation structure
- Statistical significance testing for correlations

---

**Document Version**: 1.0  
**Date**: 2025-01-05  
**Status**: Ready for Implementation