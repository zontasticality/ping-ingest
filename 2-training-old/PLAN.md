# Train a Latency Prediction Transformer

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

