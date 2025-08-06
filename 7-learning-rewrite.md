# Network Latency Prediction: Temporal Context Ablation Study

## Research Question
**Does incorporating timestamp information improve network latency prediction accuracy?**

## Objective
Train a transformer model to predict network latency, then compare performance with and without temporal information using conditional masking to isolate the temporal contribution.

## Methodology

### Conditional Temporal Ablation
- **Single model architecture** that learns to handle both conditions
- **Condition tokens**: `<TEMPORAL>` or `<NO_TEMPORAL>` explicitly tell model whether to use timestamps
- **Training**: Model learns to ignore timestamps when seeing `<NO_TEMPORAL>` token
- **Evaluation**: Same model tested with both condition tokens to measure temporal benefit

### Input Sequence Structure (IPv4-Only)
```
<TEMPORAL>
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, SUCCESS, timestamp_emb, latency_emb] <MEAS_SEP>
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, FAIL, timestamp_emb, latency_emb] <MEAS_SEP>
...
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, ?, timestamp_emb] <- PREDICT

<NO_TEMPORAL>  
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, SUCCESS, timestamp_emb, latency_emb] <MEAS_SEP>
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, FAIL, timestamp_emb, latency_emb] <MEAS_SEP>
...
[src_byte_1, src_byte_2, src_byte_3, src_byte_4, dst_byte_1, dst_byte_2, dst_byte_3, dst_byte_4, ?, timestamp_emb] <- PREDICT
```

### Token Counts (IPv4-Only)
- **IPv4 measurement**: 10 discrete tokens + 2 continuous embeddings (timestamp, latency)
- **Vocabulary**: 262 discrete tokens (0-255 bytes, SUCCESS, FAIL, MEAS_SEP, TEMPORAL, NO_TEMPORAL)
- **Maximum sequence**: 128 measurements × 10 tokens = ~1,280 tokens

### Prediction Tasks
- **Success prediction**: Will the ping succeed? (SUCCESS/FAIL)
- **Latency regression**: What will the latency be? (if successful)

## Dataset

### Scale (IPv4-Only Filter Applied)
- **Size**: ~464GB, 964 parquet files, 26.2 billion IPv4-only ping measurements (96.1% retention)
- **Timespan**: 1 week continuous measurements
- **Network**: ~20,000 probe nodes measuring latency to IPv4 destinations
- **Path**: `data/ping_super_optimized_fixed/part_0001.parquet` through `part_0964.parquet`
- **Filtering**: Only measurements where both source and destination are IPv4 addresses

### Schema (IPv4-Only)
```
prb_id: uint32          # Probe identifier (source node)
ts: int64               # Unix timestamp
sent: uint8             # Packets sent
rcvd: uint8             # Packets received
avg: float32            # Average latency (ms) - target variable
src_ipv4_int: uint32    # Source IPv4 as integer
dst_ipv4_int: uint32    # Destination IPv4 as integer
```

**Note**: IPv6-related columns (dst_is_ipv6, dst_ipv6_bytes, src_is_ipv6, src_ipv6_bytes) are filtered out during preprocessing.

## Sampling Strategy

### Source Node Selection
Focus on probes with sufficient temporal context:
- **Minimum activity**: ≥20 measurements per probe
- **Destination diversity**: ≥5 unique destinations per probe
- **Result**: ~60% of dataset from high-quality nodes

### Temporal Window Sampling
Sample sequences of different lengths to capture various temporal patterns:
- **Short sequences**: 20-40 measurements (covering seconds to minutes)
- **Medium sequences**: 40-80 measurements (covering minutes to tens of minutes)
- **Long sequences**: 80-128 measurements (covering tens of minutes to hours)

### Training Data Generation
```python
def generate_sequences(probe_data):
    # Sort measurements by timestamp for each probe
    probe_data = probe_data.sort('ts')
    
    # Extract overlapping windows of various sizes
    for window_size in [20, 40, 80, 128]:
        for start_idx in range(0, len(probe_data) - window_size, window_size // 2):
            sequence = probe_data[start_idx:start_idx + window_size]
            
            # Generate both conditions from same sequence
            for condition in ['<TEMPORAL>', '<NO_TEMPORAL>']:
                # Add condition token at start of sequence
                sequence_with_condition = [condition] + sequence
                yield sequence_with_condition
```

### Adaptive Training Duration
- **Time limit**: Stop after 6-48 hours (configurable)
- **Convergence**: Stop if loss improvement < 0.1% for 1000 steps
- **Condition validation**: Monitor that temporal vs non-temporal predictions remain distinct

## Model Architecture

### Transformer Specification (Optimized for IPv4)
- **Type**: Decoder-only (causal attention)
- **Layers**: 6 transformer blocks
- **Hidden dimension**: 512
- **Attention heads**: 8
- **Parameters**: ~25M total (reduced due to shorter sequences)

### Embedding Strategy
```python
class HybridEmbedding(nn.Module):
    def __init__(self):
        self.token_embedding = nn.Embedding(262, 512)  # Discrete tokens (IPv4-only)
        self.timestamp_projection = nn.Linear(1, 512)  # Unix timestamp
        self.latency_projection = nn.Linear(1, 512)    # Latency value
        self.positional_encoding = SinusoidalPositionalEncoding(512)
    
    def forward(self, tokens, timestamps, latencies, positions):
        # Project all to same 512-dim space
        token_emb = self.token_embedding(tokens)
        time_emb = self.timestamp_projection(timestamps.unsqueeze(-1))  
        latency_emb = self.latency_projection(latencies.unsqueeze(-1))
        pos_emb = self.positional_encoding(positions)
        
        # Combine embeddings (timestamps/latencies only at appropriate positions)
        combined = token_emb + pos_emb
        combined[timestamp_positions] += time_emb[timestamp_positions]
        combined[latency_positions] += latency_emb[latency_positions]
        
        return combined
```

### Output Heads
- **Success classifier**: `nn.Linear(512, 2)` → [FAIL_prob, SUCCESS_prob]
- **Latency regressor**: `nn.Linear(512, 1)` → predicted latency
- **Loss function**: `BCE(success) + MSE(latency | success)`

### Training Configuration (IPv4-Optimized)
- **Hardware**: Single A100 40GB (~3GB memory usage)
- **Batch size**: 128-256 sequences (larger due to shorter sequences)
- **Optimizer**: AdamW with gradient clipping
- **Learning rate**: 1e-4 with cosine decay and warmup

## Evaluation

### Data Splits
- **Training probes**: 70% of probes (all their data from days 1-5)
- **Validation**: Same training probes (day 6 data)
- **Test probes**: 30% held-out probes (day 7 data)

### Metrics
Compare model performance under both conditions:
- **Primary**: Δ MAE (reduction in mean absolute error from temporal context)
- **Secondary**: Δ accuracy for success prediction
- **Statistical**: Confidence intervals and significance tests

### Baselines
- **Persistence**: Last observed latency for each probe-destination pair
- **Moving average**: 7-measurement window average
- **Linear trend**: Extrapolate recent latency trend

### Analysis Plan
1. **Overall temporal benefit**: Average improvement across all test sequences
2. **Sequence length effects**: Does temporal benefit increase with longer context?
3. **Network topology effects**: Does benefit vary with IP address patterns?
4. **Temporal pattern analysis**: What time intervals provide most predictive value?

## Expected Results

### Hypotheses
1. **Network structure dominates**: IP addresses encode most latency information
2. **Temporal refinement**: Timestamps provide modest but measurable improvement
3. **Context-dependent benefit**: Longer sequences show greater temporal benefit

### Success Criteria
**Minimal**: Statistically significant quantification of temporal benefit (positive, negative, or neutral)

**Ideal**: Understanding of when temporal context helps, with practical recommendations for network monitoring

## Implementation

### Data Pipeline (IPv4-Only)
```python
def train_model(target_hours=12):
    model = LatencyTransformer()
    
    # Filter to high-quality probes with IPv4-only measurements
    good_probes = get_probes_with_sufficient_ipv4_data()
    
    start_time = time.time()
    for probe_id in good_probes:
        # Load and filter to IPv4-only measurements
        probe_data = load_probe_data_ipv4_only(probe_id)
        
        for condition, sequence in generate_sequences(probe_data):
            batch = create_batch(condition, sequence)
            loss = train_step(model, batch)
            
            if time.time() - start_time > target_hours * 3600:
                break
                
        if loss_converged() or time_exceeded():
            break
```

### Monitoring
- **TensorBoard**: Real-time loss and metric tracking
- **Condition validation**: Verify model learns to use condition tokens
- **Checkpointing**: Hourly model saves for recovery