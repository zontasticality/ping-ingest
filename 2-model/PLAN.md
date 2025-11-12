# Train a Latency Prediction Transformer

Train a decoder-only transformer
 - Written in JAX + Equinox + Heliax so it is comprehensible. Highly modular across multiple files and well-commented.
 - Transformer should learn various associations between ips and latency, either predicting latency or ip.

The token structure is as follows:
```rust
enum Token {
	MeasurementStart, // denotes a measurement start 
	MeasurementEnd, // denotes a measurement end
	SrcIpStart, // Marks a src ip start
	DestIpStart, // Marks a dest ip start
	Ipv4Start, // Marks an ipv4 start
	Ipv6Start, // marks an Ipv6 start
	Short(u16), // encodes a 2-byte sequence as a single token. (for ip addrs, 2 for v4, 8 for v6)
	Number(f64), // encodes an arbitrary floating point number (can be a latency, bandwidth)
	Time(u64), // encodes a time, focusing non-log embeddings.
	LatencyStart,
	Latency(u64), // Represents latency measured in millis, log-scale encoded using a separate learned matrix
	ThroughputStart,
	Throughput(u64), // Represents bandwidth measured in bytes/second, log-scale, encoded using a learned matrix
	TimestampStart,
	Timestamp(u64), // Represents the timestamp, also encoded via fourier embedding
}
```

Dataset is randomly sampled to get a list of measurements sorted by timestamp.
Each measurement has the following format:
 - `<Ip> := <Ipv4Start><Short><Short>|<Ipv6Start> 8*<Short>`
 - `<SrcIp> := <SrcIpStart><Ip>`
 - `<DestIp> := <DestIpStart><Ip>`
 - `<MeasurementStart>` // then some random ordering of SrcIp, DestIp, and LatencyStart, optionally TimestampStart and ThroughputStart

TODO: Timestamp-based positional embedding for measurements

Train the transformer on as much of the dataset as possible.
The randomization of measurement order now means the transformer should have learned various prediction modes:
 - Src+Dst -> Lat (latency prediction)
 - Src+Lat -> Dst (node within some latency range)
 - PartialIp -> Ip (allocation structure of IPs)

Can now use this to sample for paths directly of some latency.
Potentially future ability to predict cost of paths once running in real network and trained in a distributed fashion.

On: RIPE Atlas Latency Data
 - Duration: 1 month
 - Content: Measurements between Anchor nodes
   - Source IP(ipv4/6)
   - Dest IP (ipv4/6)
   - Optional timestamp (unix w/ second precision)
   - 1 or more latency measurements (milliseconds w/ floating point precision)

How:
 - Given a sequence of measurements (src/dest ip, timestamp, measurements), predict distribution over next measurement latency between given src and dst ip (at optionally given timestamp)

For What Purpose?:
 - Enable Peer-to-Peer networks to estimate latency between arbitrary nodes on the internet.


# Model Architecture



# Data Processing

We have a schema of the existing data as follows:
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

# Training Details

