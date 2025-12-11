# Train a Latency Prediction Transformer

Train a decoder-only transformer
 - Written in JAX + Equinox + Heliax so it is comprehensible. Highly modular across multiple files and well-commented.
 - Transformer should learn various associations between ips and latency, either predicting latency or ip.

## Tokenization

The token structure is as follows:
```rust
enum Token {
	MeasurementStart, // denotes a measurement start
	SrcIpStart, // Marks a src ip start
	DestIpStart, // Marks a dest ip start
	Ipv4Start, // Marks an ipv4 start
	Ipv6Start, // marks an Ipv6 start
	LatencyStart, // marks a latency measurement start
	ThroughputStart, // marks a throughput measurement start
	TimestampStart, // marks a timestamp start

	// unified bytes token to represent everything from ip addresses to timestamps to exponents
	Byte0,
	// {...}
	Byte255,
}
```

### Token Format

Dataset is randomly sampled to get a list of measurements sorted by timestamp.
Each measurement has the following format:
 - `<U8> := <Byte0> | <Byte1> | ... | <Byte255>`
 - `<Ip> := (<Ipv4Start>4*<U8>) | (<Ipv6Start> 16*<U8>)`
 - `<SrcIp> := <SrcIpStart><Ip>`
 - `<DestIp> := <DestIpStart><Ip>`
 - 
 - `<PositiveFloat> := <U8> <U8>` // encodes exponent, then mantissa, each as a single byte.
 - `<Latency> := <LatencyStart><PositiveFloat>`
 - `<Throughput> := <ThroughputStart><PositiveFloat>`
 - `<Timestamp> := 8*<u8>` // Encodes timestamp big-endian (should be milliseconds since unix epoch)
 - `<Measurement> := <MeasurementStart> permutations(<SrcIP>, <DestIp>, <Latency>, <Throughput>, <Timestamp>)`

### Rationale

The randomization of measurement properties within each measurement now means the causal decoder-only transformer needs to learn various prediction modes, e.g.:
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

