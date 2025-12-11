import duckdb
import os
import time

INPUT_GLOB = "data/parquet_ping/*.parquet"
OUTPUT_FILE = "data/training_set/training_data.parquet"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Limit RAM usage to prevent crash
MEMORY_LIMIT = "16GB"
# Directory for spillover (Ensure you have ~50GB free space here)
TEMP_DIR = "data/duckdb_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_ROWS = 100_000_000


def generate_memory_safe_sample():
    con = duckdb.connect()

    # --- CRITICAL MEMORY SETTINGS ---
    print(f"Setting Memory Limit to {MEMORY_LIMIT}...")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    # Threads: Limit threads if your CPU usage is also causing instability
    # con.execute("SET threads=4")

    print(f"--- Step 1: Scanning Statistics ---")
    # Identify Connection Tails (Failures) vs Latency Tails
    stats_query = f"""
    SELECT 
        count(*) as total,
        SUM(CASE 
            WHEN packet_error_count > 0 THEN 1  -- Packet Loss (Connection Tail)
            WHEN dup > 0 THEN 1                 -- Duplicates (Network loop/anomaly)
            ELSE 0 
        END) as conn_tail_rows
    FROM '{INPUT_GLOB}'
    """
    print("Counting rows...")
    stats = con.execute(stats_query).fetchone()
    total_rows = stats[0]
    bad_rows = stats[1]  # "Bad" now means Connection Failure/Anomaly
    good_rows = total_rows - bad_rows

    print(
        f"Total: {total_rows:,} | Connection Tails: {bad_rows:,} | Normal: {good_rows:,}"
    )

    # Calculate Probabilities
    if bad_rows >= TARGET_ROWS:
        p_bad = TARGET_ROWS / bad_rows
        p_good = 0.0
    else:
        remaining = TARGET_ROWS - bad_rows
        p_bad = 1.0
        p_good = remaining / good_rows if good_rows > 0 else 0

    print(f"Sampling Rates -> Bad: {p_bad:.4f}, Good: {p_good:.4f}")

    print(f"\n--- Step 2: Streaming & Sorting (With Disk Spill) ---")

    final_query = f"""
    COPY (
        SELECT * EXCLUDE (is_bad)
        FROM (
            SELECT *,
                CASE 
                    WHEN packet_error_count > 0 OR dup > 0 THEN 1 
                    ELSE 0 
                END as is_bad
            FROM '{INPUT_GLOB}'
        )
        WHERE 
            (is_bad = 1 AND random() < {p_bad})
            OR
            (is_bad = 0 AND random() < {p_good})
            
        -- The Sort: This will now SPILL TO DISK if it exceeds 16GB RAM.
        -- It will be slower than RAM, but it will not crash.
        ORDER BY event_time ASC
    ) 
    TO '{OUTPUT_FILE}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
    """

    try:
        start = time.time()
        con.execute(final_query)
        print(f"Success! Taken: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    generate_memory_safe_sample()
