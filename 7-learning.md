# Network Latency Prediction with Transformers

## Objective
Train a transformer model to predict network latency between arbitrary node pairs using global network measurement context.

## Problem Formulation

### Input Sequence
```
[SRC_TYPE_1, src_bytes_1..., DST_TYPE_1, dst_bytes_1..., timestamp_emb_1, SUCCESS/FAIL_1, (latency_emb_1), <MEAS_SEP>,
 SRC_TYPE_2, src_bytes_2..., DST_TYPE_2, dst_bytes_2..., timestamp_emb_2, SUCCESS/FAIL_2, (latency_emb_2), <MEAS_SEP>,
 ...,
 SRC_TYPE_target, src_bytes_target..., DST_TYPE_target, dst_bytes_target..., timestamp_emb_target]
```

### Training Targets
- **Success prediction**: Binary classification at each measurement boundary
- **Latency prediction**: Regression (only when success is predicted)
- **Prediction point**: At target measurement position

### Sequence Parameters
- **Context window**: 128 measurements (reduced for memory efficiency)
- **IPv4 sequence**: 11 tokens per measurement (5 src + 5 dst + timestamp + status + optional latency + separator)
- **IPv6 sequence**: 35 tokens per measurement (17 src + 17 dst + timestamp + status + optional latency + separator)
- **Maximum sequence length**: ~1400 tokens (128 × 11 for IPv4)
- **Prediction horizon**: Next measurement for given IP pair

## Model Architecture (50M Parameters)

### Token & Value Representation
- **Discrete vocabulary**: 261 tokens (IPv4_TYPE, IPv6_TYPE, 256 byte values, SUCCESS, FAIL, MEAS_SEP)
- **IP encoding**: 
  - IPv4: `[IPv4_TYPE, b1, b2, b3, b4]` (5 tokens)
  - IPv6: `[IPv6_TYPE, b1, b2, ..., b16]` (17 tokens)
- **Continuous values**:
  - Timestamp: Normalized within sequence window, linear projection to embedding space
  - Latency: Log-scaled then normalized, linear projection to embedding space
- **Sequence boundaries**: MEAS_SEP token separates individual measurements

### Hybrid Embedding Architecture
- **Token embedding**: nn.Embedding(261, 192) for discrete tokens
- **Timestamp projection**: nn.Linear(1, 192) for normalized timestamps
- **Latency projection**: nn.Linear(1, 192) for log-scaled latencies
- **Combination strategy**: Add token, timestamp, and latency embeddings at appropriate positions
- **Positional encoding**: Sinusoidal encoding for sequence order

### Transformer Architecture
- **Type**: Decoder-only (causal attention like GPT)
- **Layers**: 6 transformer decoder layers
- **Hidden dimension**: 768
- **Attention heads**: 12
- **Attention mask**: Causal mask preventing attention to future measurements
- **Total parameters**: ~50M

### Multi-Task Output Heads
- **Success classifier**: nn.Linear(192, 2) → [fail_prob, success_prob]
- **Latency regressor**: nn.Linear(192, 1) → predicted latency value
- **Training loss**: α × BCE(success) + β × MSE(latency | success)
- **Loss weights**: α = 1.0, β = 1.0 (balanced multi-task learning)

## Training Strategy

### Data Preparation
1. Extract overlapping sequences from parquet files
2. Sort by timestamp globally
3. Create training pairs with various prediction horizons

### Hardware Requirements
- **Target**: Single A100 40GB GPU
- **Training time**: 2 hours maximum
- **Memory usage**: ~4-5GB GPU memory

### Training Configuration
- **Dataset**: 50% sample (3-4B measurements) from 1 week
- **Batch size**: 512-1024
- **Learning rate**: 1e-4 with cosine decay and warmup
- **Optimization**: AdamW with gradient clipping

### Training Configuration
- **Data sampling**: 50% random sample from 1 week (~3-4B measurements)
- **Sequence creation**: Sliding window with 50% overlap
- **Failed ping definition**: sent ≠ rcvd OR avg ≤ 0 OR avg > 10000ms
- **Batch strategy**: Dynamic batching with max 8192 tokens per batch

### Training Objectives & Monitoring
- **Primary loss**: Hybrid success classification + conditional latency regression
- **Key metrics**: 
  - Success accuracy, F1-score
  - Latency MAE, RMSE
  - Latency accuracy within ±10ms, ±50ms thresholds
  - Attention entropy (to detect attention collapse)

### Evaluation Strategy
- **Temporal split**: Days 1-5 train, day 6 validation, day 7 test
- **Node generalization**: Test on unseen IP pairs
- **Baseline comparisons**: 
  - Persistence model (last observed latency)
  - Moving average (7-measurement window)
  - Linear regression on recent measurements
- **Success criteria**:
  - Success prediction accuracy >85%
  - Latency MAE <20ms for 5-minute predictions
  - >60% of predictions within ±10ms of actual

## Evaluation Metrics

### Prediction Quality
- RMSE vs prediction horizon (1min, 5min, 15min, 60min)
- Percentage within ±X ms of actual latency
- Baseline comparison: persistence, moving average

### Interpretability Analysis
- **Attention patterns**: Which context measurements influence predictions
- **IP byte importance**: How model processes IP address structure
- **Temporal dependencies**: Effective context window analysis
- **Geographic correlations**: Attention weights vs network topology

## Expected Insights & Research Value

### Network Predictability Analysis
- **Upper bound assessment**: Maximum achievable prediction accuracy with full network visibility
- **Temporal patterns**: How far into the future can network latency be reliably predicted?
- **Context dependencies**: Which measurements in the sequence are most informative for prediction?

### Learned Network Representations
- **IP address structure**: How does the model process and relate different network addresses?
- **Failure pattern recognition**: What network conditions precede measurement failures?
- **Attention patterns**: Which context measurements influence predictions most strongly?

### Scientific Contributions
- **Modelability benchmark**: Establish performance ceiling for network latency prediction
- **Feature importance**: Quantify the value of different types of network context
- **Failure prediction**: Demonstrate feasibility of proactive network monitoring

## Implementation Plan

### Phase 1: Data Pipeline Development
- **Streaming processing**: Process parquet files sequentially to manage memory
- **Preprocessing pipeline**: 
  - Convert IP addresses to byte sequences
  - Normalize timestamps within sequence windows
  - Log-scale and normalize latency values
  - Identify failed pings using defined criteria
- **Sequence generation**: Sliding window approach with overlap
- **Dynamic batching**: Pack sequences efficiently to maximize GPU utilization

### Phase 2: Model Implementation
- **Hybrid architecture**: Implement mixed token embedding + continuous value projection
- **Causal transformer**: Decoder-only architecture with proper attention masking
- **Multi-task heads**: Separate success classification and latency regression outputs
- **Training infrastructure**: 
  - Gradient clipping for stability
  - Learning rate warmup and cosine decay
  - Model checkpointing with early stopping
  - Weights & Biases integration for monitoring

### Phase 3: Training & Evaluation
- **Training execution**: 2-hour training run on A100 40GB with progress monitoring
- **Model analysis**:
  - Attention pattern visualization (which context measurements matter most)
  - IP byte importance analysis (how model processes address structure)
  - Temporal dependency analysis (effective context window)
- **Performance evaluation**: Comprehensive comparison against baseline methods
- **Scientific insights**: Network predictability patterns and learned representations

## Success Criteria & Risk Mitigation

### Technical Success Metrics
- **Prediction accuracy**: Latency MAE <20ms for next measurement prediction
- **Success classification**: >85% accuracy in predicting ping success/failure  
- **Generalization**: Performance maintained on unseen IP pairs (temporal holdout)
- **Efficiency**: Training completes within 2-hour target on single A100

### Scientific Success Metrics
- **Interpretability**: Attention patterns reveal meaningful network relationships
- **Predictability bounds**: Establish quantitative limits of network latency forecasting
- **Context analysis**: Identify optimal sequence length for prediction tasks

### Risk Mitigation Strategies
- **Memory overflow**: Dynamic batching with token limits, gradient checkpointing if needed
- **Training instability**: Gradient clipping, learning rate warmup, early stopping
- **Poor convergence**: Learning rate tuning, architecture adjustments (fewer layers if needed)
- **Insufficient data quality**: Fallback to simpler regression baseline if transformer underperforms