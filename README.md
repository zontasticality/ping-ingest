# Ping-Ingest

Ingest raw RIPE Atlas ping datasets and convert them into optimized Parquet files for network simulation and analysis.

## Overview

This project provides a data pipeline for processing large-scale ping measurement data from RIPE Atlas. It converts compressed JSON files (`.bz2`) into time-sorted Parquet files with efficient compression, then supports aggregation, sampling, and analysis workflows.

## Data Pipeline

```
Raw Data (bz2) → Parse (JSON→Parquet) → Aggregate/Sample → Analysis
```

1. **Download**: Raw ping measurements from RIPE Atlas (`.json.bz2` files)
2. **Parse**: Convert to Parquet with time-based sorting and ZSTD compression
3. **Aggregate**: Generate training datasets with uniform random sampling
4. **Analyze**: Inspect data quality, statistics, and distributions

## Scripts

### Core Pipeline Scripts

#### `duckdb_parallel_parse.py`
**Purpose**: Parallel parser for converting compressed bz2 ping data to Parquet files.

**Features**:
- Random or sequential file selection
- Parallel processing using multiple CPU cores
- Automatic temp file cleanup
- Progress tracking with timestamps
- Configurable memory limits and temp directory location

**Usage**:
```bash
# Process 10 random files with 8 workers
python3 duckdb_parallel_parse.py -n 10 -w 8

# Process all files in randomized order (default)
python3 duckdb_parallel_parse.py

# Process all files sequentially (no randomization)
python3 duckdb_parallel_parse.py --no-random

# Dry run to see which files would be selected
python3 duckdb_parallel_parse.py -n 20 --dry-run

# Use local /tmp for temp files (recommended for NFS)
python3 duckdb_parallel_parse.py --local-tmp
```

**Options**:
- `-n, --num-files`: Number of files to process (default: all)
- `-w, --workers`: Number of parallel workers (default: min(8, CPU count))
- `--no-random`: Disable random selection, process in sorted order
- `--dry-run`: Show selected files without processing
- `--local-tmp`: Use /tmp for temp files instead of network filesystem

**Output**: `data/parquet_ping/*.parquet` - Time-sorted Parquet files with ZSTD compression

---

#### `duckdb_parse_json.py`
**Purpose**: Simple single-threaded parser for bz2 JSON to Parquet conversion.

**Note**: This is a simpler, non-parallel version. Use `duckdb_parallel_parse.py` for better performance.

**Usage**:
```bash
python3 duckdb_parse_json.py
```

**Input**: `data/bz2_ping/*.json.bz2`
**Output**: `data/parquet_ping/*.parquet`

---

#### `duckdb_aggregate.py`
**Purpose**: Generate training datasets by aggregating and sampling Parquet files.

**Features**:
- Uniform random sampling across all input files
- Configurable target row count (default: 100M rows)
- Memory-efficient processing with external sorting
- Preserves temporal ordering from input files

**Usage**:
```bash
python3 duckdb_aggregate.py
```

**Input**: `data/parquet_ping/*.parquet`
**Output**: `data/training_set/training_data.parquet` (100M row sample)

**Schema** (minimal training format):
- `msm_id`: Measurement ID
- `event_time`: Timestamp
- `src_addr`, `dst_addr`: IP addresses
- `ip_version`: IPv4/IPv6 indicator
- `rtt`: Round-trip time (float32)
- `size`: Packet size (float32)
- `packet_error_count`: Error count

---

### Analysis and Utility Scripts

#### `duckdb_sample.py`
**Purpose**: Diagnostic tool for inspecting training data quality and sampling sequences.

**Features**:
- Check time range coverage
- Verify dataset duration
- Generate truly random sample sequences

**Usage**:
```bash
python3 duckdb_sample.py
```

**Input**: `data/training_set/minimal_training_data.parquet`

---

#### `duckdb_stats.py`
**Purpose**: Analyze destination address statistics and cardinality.

**Features**:
- Check sparsity and cardinality
- Identify most common destinations
- Measure dictionary encoding effectiveness

**Usage**:
```bash
python3 duckdb_stats.py
```

**Input**: `data/training_set/training_data.parquet`

---

#### `duckdb_size.py`
**Purpose**: Inspect Parquet file column sizes and compression ratios.

**Features**:
- Per-column compressed/uncompressed sizes
- Compression ratio analysis
- Identify columns with poor compression
- Calculate percentage of disk space per column

**Usage**:
```bash
python3 duckdb_size.py
```

**Input**: `data/training_set/training_data.parquet`

**Example Output**:
```
column_name       compressed_mb  uncompressed_mb  compression_ratio  pct_of_disk
dst_addr                 45.23           123.45                2.7         35.2
event_time               28.91            98.12                3.4         22.5
```

---

## Directory Structure

```
ping-ingest/
├── data/
│   ├── raw_ping/           # Raw .bz2 files from RIPE Atlas
│   ├── parquet_ping/       # Parsed Parquet files (time-sorted)
│   ├── training_set/       # Aggregated training datasets
│   ├── temp_decomp/        # Temporary decompression files
│   └── duckdb_tmp/         # DuckDB external sort temp files
├── duckdb_parallel_parse.py    # Main parallel parser
├── duckdb_parse_json.py        # Simple single-threaded parser
├── duckdb_aggregate.py         # Training data aggregation
├── duckdb_sample.py            # Data quality diagnostics
├── duckdb_stats.py             # Destination address analysis
├── duckdb_size.py              # Column size analysis
└── old_processing/             # Legacy Jupyter notebooks
```

## Typical Workflow

1. **Download raw data** to `data/raw_ping/`

2. **Parse to Parquet** (parallel processing):
   ```bash
   # Process all files in random order
   python3 duckdb_parallel_parse.py
   ```

3. **Generate training dataset** (100M row sample):
   ```bash
   python3 duckdb_aggregate.py
   ```

4. **Analyze output**:
   ```bash
   # Check file sizes and compression
   python3 duckdb_size.py

   # Verify data quality
   python3 duckdb_sample.py

   # Inspect destination statistics
   python3 duckdb_stats.py
   ```

## Performance Notes

- **Parallel Parsing**: Uses `bunzip2` for decompression, parallelized at the file level (8 concurrent files by default)
- **Memory Management**: Configurable memory limits (default: 24GB for parsing, 16GB for aggregation)
- **Compression**: All Parquet files use ZSTD codec with 100K row groups
- **Sorting**: Time-sorted output enables efficient temporal queries
- **Random Selection**: When processing all files, default behavior randomizes order for better distribution if interrupted

## Requirements

- Python 3.12+
- DuckDB 0.8.0+
- bunzip2 (for decompression)
- Sufficient disk space for intermediate files

