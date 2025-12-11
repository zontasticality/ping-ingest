# Network Latency Prediction: Timestamp Context Ablation Study

## Research Question
**How much does incorporating timestamp information improve network latency prediction accuracy?**

## Objective
Train a single transformer model with conditional timestamp masking to compare temporal vs non-temporal predictions using identical architecture and parameters.

## Methodology

### Input Sequence Structure
```
<CONDITION_TOKEN> <RESOLUTION_TOKEN>
[SRC_TYPE_1, src_bytes_1..., DST_TYPE_1, dst_bytes_1..., timestamp_emb_1, SUCCESS/FAIL_1, latency_emb_1] <MEAS_SEP>
[SRC_TYPE_2, src_bytes_2..., DST_TYPE_2, dst_bytes_2..., timestamp_emb_2, SUCCESS/FAIL_2, latency_emb_2] <MEAS_SEP>
...
[SRC_TYPE_target, src_bytes_target..., DST_TYPE_target, dst_bytes_target..., timestamp_emb_target]
```

### Conditional Ablation Strategy
- **Condition tokens**: `<TEMPORAL>` and `<NO_TEMPORAL>` control timestamp usage
- **Resolution tokens**: `<SECOND>`, `<MINUTE>`, `<HOUR>`, `<DAY>` indicate temporal scale
- **Training**: Model learns when to use temporal information based on explicit tokens
- **Evaluation**: Same model tested with both conditions to isolate temporal contribution

### Prediction Tasks
- **Success prediction**: Will the target measurement succeed? (`sent == rcvd`)
- **Latency regression**: Predicted latency value (if successful)

## Dataset

### Scale & Structure
- **Size**: 483GB, 964 parquet files, 27.2 billion ping measurements
- **Timespan**: 1 week continuous measurements (0.3s average intervals)
- **Network**: ~20,000 probe nodes measuring latency to multiple destinations
- **File path**: `data/ping_super_optimized_fixed/part_0001.parquet` through `part_0964.parquet`

### Schema
```
prb_id: uint32          # Probe identifier
ts: int64               # Unix timestamp
sent: uint8             # Packets sent
rcvd: uint8             # Packets received
avg: float32            # Average latency (ms) - target variable
rtt_1,2,3: float32      # Individual round-trip times
dst_is_ipv6: bool       # Destination IP version
dst_ipv4_int: uint32    # Destination IPv4 as integer
dst_ipv6_bytes: binary  # Destination IPv6 as bytes
src_is_ipv6: bool       # Source IP version
src_ipv4_int: uint32    # Source IPv4 as integer
src_ipv6_bytes: binary  # Source IPv6 as bytes
dst_addr_display: str   # Human-readable destination IP
src_addr_display: str   # Human-readable source IP
```

### Characteristics
- **Latency**: Mean 92ms, std 98ms, range -1ms to 51,083ms
- **Success rate**: ~95% (sent == rcvd)
- **Node activity**: Highly skewed (1 to 1.25M measurements per probe)

## Sampling Strategy

### Multi-Resolution Temporal Windows
Extract temporal patterns at multiple scales:

- **Second-level**: 10s windows, 1s stride (micro-bursts)
- **Minute-level**: 5min windows, 30s stride (load patterns)  
- **Hour-level**: 2h windows, 15min stride (daily cycles)
- **Day-level**: 1day windows, 4h stride (weekly patterns)

### Source Node Filtering
Train only on probes with sufficient temporal context:
- **Minimum**: ≥20 measurements and ≥5 destinations per probe
- **Coverage**: ~60% of dataset from high-quality nodes

### Adaptive Training
- **Duration**: Stop after time limit (6-48h) or loss convergence
- **Rationale**: IP structure enables efficient learning of network topology

## Model Architecture

### Token Representation (~57M Parameters)
- **Vocabulary**: 267 tokens (IP bytes, condition/resolution tokens, separators)
- **IP encoding**: IPv4 as 4 bytes, IPv6 as 16 bytes
- **Continuous values**: Timestamps and latencies projected to embedding space
- **Architecture**: 6-layer decoder-only transformer (768 dim, 12 heads)

### Training Configuration
- **Hardware**: Single A100 40GB (~6GB memory usage)
- **Batch size**: 64-128 sequences (dynamic based on length)
- **Optimization**: AdamW with gradient clipping, cosine decay
- **Loss**: Multi-task BCE(success) + MSE(latency | success)

## Evaluation

### Data Splits
- **Training**: 70% of probes (days 1-5)
- **Validation**: Training probes (day 6)
- **Testing**: 30% held-out probes (day 7)

### Metrics
- **Primary**: Δ MAE and Δ accuracy between temporal/non-temporal conditions
- **Baselines**: Persistence, moving average, linear regression
- **Analysis**: Multi-scale temporal benefits, network topology effects

### Statistical Testing
Confidence intervals and significance tests for temporal improvements across different:
- Time scales (second/minute/hour/day)
- Network characteristics (probe types, IP similarities)
- Traffic patterns (load levels, geographic factors)

## Expected Outcomes

### Hypotheses
1. **Network structure learning**: Model efficiently learns IP topology patterns
2. **Scale-dependent benefits**: Temporal context helps more at certain time scales
3. **Conditional improvement**: Benefits vary with network characteristics

### Research Value
- **Empirical evidence**: Multi-scale temporal analysis of network latency prediction
- **Practical guidance**: Inform network monitoring system temporal feature decisions
- **Methodological contribution**: Condition-token ablation for network modeling

## Technical Implementation

### Data Pipeline
```python
def create_training_data(target_hours=12):
    good_probes = filter_probes_by_activity()
    
    for resolution in ['second', 'minute', 'hour', 'day']:
        for probe in good_probes:
            sequences = extract_temporal_windows(probe, resolution)
            for seq in sequences:
                yield add_condition_tokens(seq, resolution)
    
    if training_time_exceeded(target_hours) or loss_plateaued():
        break
```

### Monitoring
- **Real-time**: TensorBoard logging
- **Validation**: Condition token effectiveness monitoring  
- **Checkpointing**: Hourly saves for recovery

## Success Criteria

**Minimal**: Clear quantification of temporal benefit with statistical significance.

**Ideal**: Understanding of when, where, and at what scales temporal context improves network prediction, with practical recommendations for monitoring systems.