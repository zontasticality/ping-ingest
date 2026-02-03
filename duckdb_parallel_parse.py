#!/usr/bin/env python3
"""
Parallel DuckDB parser for bz2-compressed ping data.

Features:
- Random file selection from data/raw_ping/*.bz2
- Parallel decompression using bunzip2 to temp files
- Parallel processing based on CPU core count
- Automatic temp file cleanup
- Robust error handling and progress tracking
"""

import duckdb
import os
import glob
import time
import random
import subprocess
import tempfile
import multiprocessing as mp
import signal
import atexit
from pathlib import Path
from typing import List

# Configuration
INPUT_DIR = "data/raw_ping"
OUTPUT_DIR = "data/parquet_ping"
DEFAULT_TEMP_DIR = "data/temp_decomp"  # Default temporary decompression directory
TEMP_DIR = DEFAULT_TEMP_DIR  # Will be set by main()

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cleanup_temp_files():
    """Clean up all temporary decompression files."""
    try:
        # Clean up both possible temp directories
        for temp_dir in [DEFAULT_TEMP_DIR, "/tmp/ping_decomp"]:
            if not os.path.exists(temp_dir):
                continue
            temp_files = glob.glob(os.path.join(temp_dir, "*.tmp"))
            if temp_files:
                print(f"\n\nCleaning up {len(temp_files)} temp files from {temp_dir}...")
                for tf in temp_files:
                    try:
                        os.remove(tf)
                    except:
                        pass
                print("Cleanup complete.")
    except:
        pass


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n\nReceived interrupt signal. Cleaning up...")
    cleanup_temp_files()
    exit(1)


# Register cleanup handlers
atexit.register(cleanup_temp_files)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Explicit Schema
SCHEMA_DEF = {
    "msm_id": "BIGINT",
    "prb_id": "BIGINT",
    "timestamp": "BIGINT",
    "dst_addr": "VARCHAR",
    "src_addr": "VARCHAR",
    "from": "VARCHAR",
    "dst_name": "VARCHAR",
    "min": "DOUBLE",
    "avg": "DOUBLE",
    "max": "DOUBLE",
    "rcvd": "INTEGER",
    "sent": "INTEGER",
    "dup": "INTEGER",
    "size": "INTEGER",
    "ttl": "INTEGER",
    "step": "INTEGER",
    "proto": "VARCHAR",
    "type": "VARCHAR",
    "af": "BIGINT",
    "fw": "VARCHAR",
    "mver": "VARCHAR",
    "lts": "BIGINT",
    "group_id": "BIGINT",
    "result": "STRUCT(x VARCHAR)[ ]",
}


def decompress_bz2_to_temp(bz2_path: str, temp_path: str, worker_id: int, processors: int = None) -> bool:
    """
    Decompress a bz2 file to a temporary location using bunzip2.

    Args:
        bz2_path: Path to compressed .bz2 file
        temp_path: Path where decompressed file should be written
        worker_id: Worker ID for logging
        processors: Number of processors (ignored for bunzip2, kept for compatibility)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use shell redirection for maximum I/O performance
        # Shell redirection is faster than Python's subprocess.PIPE buffering
        # k=keep original, d=decompress, c=stdout
        cmd = f'bunzip2 -kdc "{bz2_path}" > "{temp_path}"'

        decomp_start = time.time()
        print(f"[Worker {worker_id}] [{time.strftime('%H:%M:%S')}] Starting decompression with bunzip2...")

        result = subprocess.run(
            cmd,
            shell=True,
            stderr=subprocess.PIPE,
            check=True
        )

        decomp_duration = time.time() - decomp_start
        print(f"[Worker {worker_id}] [{time.strftime('%H:%M:%S')}] Decompression completed in {decomp_duration:.1f}s")

        return True

    except subprocess.CalledProcessError as e:
        print(f"[Worker {worker_id}] bunzip2 failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"[Worker {worker_id}] Decompression error: {e}")
        return False


def convert_to_parquet(input_path: str, process_id: int = 0, processors_per_worker: int = None) -> dict:
    """
    Convert a single bz2 file to parquet using temp file for decompression.

    Args:
        input_path: Path to input .bz2 file
        process_id: Worker process ID for logging
        processors_per_worker: Number of processors for pbzip2 to use

    Returns:
        dict with status, filename, duration, and error (if any)
    """
    filename = os.path.basename(input_path)
    base_name = filename.replace('.bz2', '')
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.parquet")

    # Skip if already processed
    if os.path.exists(output_path):
        return {
            "status": "skipped",
            "filename": filename,
            "duration": 0,
            "error": None
        }

    print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Processing {filename}...")
    start = time.time()

    # Create temp file path
    temp_json_path = os.path.join(TEMP_DIR, f"{base_name}.json.tmp")

    try:
        # Step 1: Decompress to temp file
        decompress_start = time.time()
        success = decompress_bz2_to_temp(input_path, temp_json_path, process_id, processors_per_worker)

        if not success:
            raise Exception("Decompression failed")

        decompress_time = time.time() - decompress_start

        # Check temp file size
        temp_size_gb = os.path.getsize(temp_json_path) / (1024**3)
        print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Decompressed to {temp_size_gb:.2f}GB in {decompress_time:.1f}s")

        # Step 2: Parse with DuckDB
        parse_start = time.time()
        print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Starting DuckDB parse...")
        con = duckdb.connect()

        # Configure DuckDB for external sorting (handles large files that don't fit in memory)
        # Use /tmp for sorting (local disk, much faster than network storage)
        con.execute("SET temp_directory='/tmp/duckdb_sort'")
        con.execute("SET memory_limit='24GB'")  # Use 24GB per worker (fits 8 workers in 120GB)
        con.execute("SET preserve_insertion_order=false")  # Faster sorting, order comes from ORDER BY
        print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Configured: memory_limit=24GB, temp_directory=/tmp/duckdb_sort")

        query = f"""
        COPY (
            SELECT
                -- 1. TIME (Primary Sort Key)
                to_timestamp(timestamp) as event_time,

                -- 2. KEYS
                msm_id, dst_addr, prb_id,

                -- 3. METRICS
                min as rtt_min, avg as rtt_avg, max as rtt_max,
                sent, rcvd, dup, size, ttl,

                -- 4. ERRORS
                len(list_filter(list_transform(result, item -> item.x), x -> x IS NOT NULL)) as packet_error_count,

                -- 5. METADATA
                src_addr, "from" as from_addr, dst_name, proto, af as ip_version,
                fw as firmware, mver as version, step, lts, group_id

            FROM read_json('{temp_json_path}', columns={SCHEMA_DEF}, format='newline_delimited')

            ORDER BY event_time ASC, msm_id
        )
        TO '{output_path}'
        (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
        """

        con.execute(query)
        con.close()

        parse_time = time.time() - parse_start
        print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Parsed to parquet in {parse_time:.1f}s")

        # Step 3: Clean up temp file
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
            print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] Cleaned up temp file")

        duration = time.time() - start
        output_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"[Worker {process_id}] [{time.strftime('%H:%M:%S')}] ✓ {filename} completed in {duration:.1f}s ({output_size_mb:.1f}MB)")

        return {
            "status": "success",
            "filename": filename,
            "duration": duration,
            "decompress_time": decompress_time,
            "parse_time": parse_time,
            "temp_size_gb": temp_size_gb,
            "output_size_mb": output_size_mb,
            "error": None
        }

    except Exception as e:
        duration = time.time() - start
        error_msg = str(e)
        print(f"[Worker {process_id}] ✗ {filename} failed: {error_msg}")

        # Clean up temp file on error
        if os.path.exists(temp_json_path):
            try:
                os.remove(temp_json_path)
                print(f"[Worker {process_id}] Cleaned up temp file after error")
            except:
                pass

        return {
            "status": "failed",
            "filename": filename,
            "duration": duration,
            "error": error_msg
        }


def worker_process(file_queue: mp.Queue, result_queue: mp.Queue, worker_id: int, processors_per_worker: int = None) -> None:
    """
    Worker process that consumes files from queue and processes them.

    Args:
        file_queue: Queue containing file paths to process
        result_queue: Queue to put results into
        worker_id: Unique worker identifier
        processors_per_worker: Number of processors for pbzip2 to use
    """
    print(f"[Worker {worker_id}] Started")

    while True:
        try:
            # Use timeout to handle timing issues
            file_path = file_queue.get(timeout=1)

            # Check for sentinel value
            if file_path is None:
                print(f"[Worker {worker_id}] Received stop signal")
                break

        except Exception:
            # Queue is empty, worker is done
            print(f"[Worker {worker_id}] Queue empty, exiting")
            break

        result = convert_to_parquet(file_path, worker_id, processors_per_worker)
        result_queue.put(result)

    print(f"[Worker {worker_id}] Finished")


def main(
    num_files: int = None,
    num_workers: int = None,
    processors_per_worker: int = None,
    random_selection: bool = True,
    dry_run: bool = False,
    use_local_tmp: bool = False
):
    """
    Main entry point for parallel parsing.

    Args:
        num_files: Number of files to process (None = all files)
        num_workers: Number of parallel workers (None = CPU count)
        processors_per_worker: [Ignored] Kept for compatibility with old API
        random_selection: Whether to randomly select files
        dry_run: Show selected files without processing
        use_local_tmp: Use /tmp (local disk) for temp files instead of network filesystem
    """
    global TEMP_DIR

    # Set temp directory location
    if use_local_tmp:
        TEMP_DIR = "/tmp/ping_decomp"
        print(f"Using LOCAL /tmp for temp files (avoids NFS bottleneck)")
    else:
        TEMP_DIR = DEFAULT_TEMP_DIR
        print(f"Using network filesystem for temp files")

    os.makedirs(TEMP_DIR, exist_ok=True)

    total_cores = mp.cpu_count()

    # Determine number of workers
    if num_workers is None:
        num_workers = min(8, total_cores)  # Default: 8 workers or total cores

    # Calculate processors per worker if not specified
    if processors_per_worker is None:
        processors_per_worker = max(1, total_cores // num_workers)

    print(f"Starting parallel parser with {num_workers} workers")
    print(f"Using bunzip2 for decompression (single-threaded per file)")
    print(f"Parallelism: {num_workers} concurrent files")

    # Get all bz2 files
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.bz2")))
    total_available = len(all_files)

    if total_available == 0:
        print(f"No .bz2 files found in {INPUT_DIR}")
        return

    print(f"Found {total_available} .bz2 files")

    # Filter out already-processed files by checking parquet_ping directory
    unprocessed_files = []
    for bz2_path in all_files:
        filename = os.path.basename(bz2_path)
        base_name = filename.replace('.bz2', '')
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.parquet")

        if not os.path.exists(output_path):
            unprocessed_files.append(bz2_path)

    already_processed = total_available - len(unprocessed_files)
    print(f"Already processed: {already_processed} files")
    print(f"Remaining to process: {len(unprocessed_files)} files")

    # Print the list of files left to parse
    if len(unprocessed_files) > 0:
        print(f"\nFiles left to parse:")
        for f in sorted(unprocessed_files):
            print(f"  {os.path.basename(f)}")
        print()

    if len(unprocessed_files) == 0:
        print("All files have already been processed!")
        return

    # Select files to process from unprocessed files
    if num_files is None:
        selected_files = unprocessed_files
        if random_selection:
            random.shuffle(selected_files)
            print(f"Processing all {len(selected_files)} unprocessed files (randomized order)")
        else:
            print(f"Processing all {len(selected_files)} unprocessed files (sorted order)")
    else:
        num_files = min(num_files, len(unprocessed_files))
        if random_selection:
            selected_files = random.sample(unprocessed_files, num_files)
            print(f"Randomly selected {num_files} files from unprocessed")
        else:
            selected_files = unprocessed_files[:num_files]
            print(f"Selected first {num_files} unprocessed files")

    # Dry run mode - just show selections
    if dry_run:
        print("\n=== DRY RUN MODE - Selected files ===")
        for i, f in enumerate(selected_files, 1):
            print(f"{i:3d}. {os.path.basename(f)}")
        print(f"\nWould process {len(selected_files)} files")
        return

    # Create queues
    file_queue = mp.Queue()
    result_queue = mp.Queue()

    # Populate file queue
    for f in selected_files:
        file_queue.put(f)

    # Start workers
    print(f"Launching {num_workers} worker processes...")
    start_time = time.time()

    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(file_queue, result_queue, i, processors_per_worker))
        p.start()
        workers.append(p)

    # Wait for all workers to complete
    for p in workers:
        p.join()

    # Collect results - we should have exactly len(selected_files) results
    results = []
    num_expected = len(selected_files)

    for _ in range(num_expected):
        try:
            result = result_queue.get(timeout=5)
            results.append(result)
        except:
            # Worker may have crashed without putting result
            break

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")

    print(f"Total files processed: {len(results)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⊘ Skipped: {skipped_count}")
    print(f"  ✗ Failed:  {failed_count}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if success_count > 0:
        avg_time = sum(r["duration"] for r in results if r["status"] == "success") / success_count
        avg_decompress = sum(r.get("decompress_time", 0) for r in results if r["status"] == "success") / success_count
        avg_parse = sum(r.get("parse_time", 0) for r in results if r["status"] == "success") / success_count

        print(f"\nAverage times per file:")
        print(f"  Total: {avg_time:.1f}s")
        print(f"  Decompress: {avg_decompress:.1f}s")
        print(f"  Parse: {avg_parse:.1f}s")

    if failed_count > 0:
        print("\nFailed files:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['filename']}: {r['error']}")

    # Clean up any remaining temp files
    temp_files = glob.glob(os.path.join(TEMP_DIR, "*.tmp"))
    if temp_files:
        print(f"\nCleaning up {len(temp_files)} orphaned temp files...")
        for tf in temp_files:
            try:
                os.remove(tf)
            except:
                pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel DuckDB parser for bz2-compressed ping data"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=None,
        help="Number of files to process (default: all)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(8, CPU count))"
    )
    parser.add_argument(
        "-p", "--processors",
        type=int,
        default=None,
        help="[Ignored - kept for compatibility] Cores per worker (bunzip2 is single-threaded)"
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Disable random file selection (process in sorted order)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be selected without processing them"
    )
    parser.add_argument(
        "--local-tmp",
        action="store_true",
        help="Use /tmp (local disk) for temp decompression instead of network filesystem (recommended for NFS)"
    )

    args = parser.parse_args()

    main(
        num_files=args.num_files,
        num_workers=args.workers,
        processors_per_worker=args.processors,
        random_selection=not args.no_random,
        dry_run=args.dry_run,
        use_local_tmp=args.local_tmp
    )
