import duckdb
import os
import glob
import time

# --- CONFIGURATION ---
INPUT_DIR = "data/parquet_ping"
OUTPUT_FILE = "data/training_set/training_data.parquet"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Memory Safety
MEMORY_LIMIT = "16GB"
TEMP_DIR = "data/duckdb_tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Target Dataset Size
TARGET_ROWS = 100_000_000


def generate_random_minimal_sample():
    con = duckdb.connect()

    # 1. Setup
    print(f"--- Configuration ---")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")

    # Get sorted list of files to preserve temporal ordering during read
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not input_files:
        print(f"No parquet files found in {INPUT_DIR}")
        return

    print(f"Found {len(input_files)} files.")

    # 2. Count Total Rows
    print(f"\n--- Step 1: Counting Total Rows ---")
    start_time = time.time()

    # Fast count using metadata
    count_query = f"SELECT count(*) FROM '{os.path.join(INPUT_DIR, '*.parquet')}'"
    total_rows = con.execute(count_query).fetchone()[0]

    print(f"Scan complete in {time.time() - start_time:.2f}s")
    print(f"Total Rows Available: {total_rows:,}")

    # 3. Calculate Uniform Sampling Rate
    if total_rows <= TARGET_ROWS:
        sample_rate = 1.0
        print(f"Target ({TARGET_ROWS:,}) > Total. Keeping 100% of data.")
    else:
        sample_rate = TARGET_ROWS / total_rows
        print(f"Target ({TARGET_ROWS:,}) < Total. Sampling rate: {sample_rate:.6f}")

    # 4. Generate Output
    print(f"\n--- Step 2: Generating Minimal f32 Parquet ---")

    files_sql_list = ", ".join([f"'{f}'" for f in input_files])

    query = f"""
    COPY (
        SELECT
            -- 1. Sequence Context
            msm_id,
            event_time,

            -- 2. IP Tokens
            from_addr,
            dst_addr,
            ip_version, 

            -- 3. Metric Tokens (f32 Precision)
            CAST(rtt_min AS FLOAT) as rtt,
            CAST(size AS FLOAT) as size,

            -- 4. Status Tokens (For Timeout logic)
            packet_error_count

        FROM read_parquet([{files_sql_list}])
        WHERE 
            -- Simple Uniform Random Sample
            random() < {sample_rate}
            
        -- We do NOT add an ORDER BY clause here.
        -- We rely on the fact that input files are sorted by time, 
        -- and DuckDB reads the list of files sequentially.
    ) 
    TO '{OUTPUT_FILE}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
    """

    try:
        write_start = time.time()
        con.execute(query)
        print(f"Success! Generation took {time.time() - write_start:.2f}s")
        print(f"Output saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    generate_random_minimal_sample()
