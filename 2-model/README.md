# Network Topology Learning via A Decoder-Only Transformer

Goal: Essentially turn measurements into a language where you have to learn the joint distribution for each measurement, a sequence of measurements.

### Tokens

```rust
enum Token {
	MeasurementStart, // denotes a measurement start 
	MeasurementEnd, // denotes a measurement end
	SrcIpStart, // Marks a src ip start
	DestIpStart, // Marks a dest ip start
	Ipv4Start, // Marks an ipv4 start
	Ipv6Start, // marks an Ipv6 start
	Short(u16), // encodes a 2-byte sequence as a single token. (for ip addrs, 2 for v4, 8 for v6)
	LatencyStart,
	Latency(u64), // Represents latency measured in millis, log-scale encoded using a separate learned matrix
	ThroughputStart,
	Throughput(u64), // Represents bandwidth measured in bytes/second, log-scale, encoded using a learned matrix
	TimestampStart,
	Timestamp(u64), // Represents the timestamp, also encoded via fourier embedding
}
```

We train on a sequence of these tokens generated from the dataset and do causal learning on it.

### Training

We have an existing RIPE Atlas dataset which contains the following data:
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
This will be processed using google's grain dataset pipelining library into a lazy-loaded stream of tokens that can be trained on.

The token generation process will take the following data from the parquet files in data/ping_super_optimized_fixed:
 - `ts` - Timestamp
 - `avg` - Average Latency of 3 measurements
 - `src_is_ipv6`, `dst_is_ipv6`
 - `dst_ipv4_int`, `src_ipv6_bytes` - IP data

This will be translated into a sequence of tokens via the following structures:

 - `<MeasurementStart>`
 - The following will be deterministically randomly permuted for each measurement
   - `<SrcIpStart>(<Ipv4Start> 2*<Short> | <Ipv6Start> 8*<Short>)`
   - `<DestIpStart>(<Ipv4Start> 2*<Short> | <Ipv6Start> 8*<Short>)`
   - `<LatencyStart><Latency>` - A negative value represents failed connection
   - `<TimestampStart><Timestamp>` - Optional
   - `<ThroughputStart><Throughput>` - Optional, no data yet
 - `<MeasurementEnd>`

### Transformer Design

This will be a standard causal attention transformer written from scratch in a comprehensible style using equinox for structure and haliax for more scrutible named matrices.

