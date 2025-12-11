import duckdb
import os
import time

# --- CONFIGURATION ---
INPUT_GLOB = "data/parquet_ping/*.parquet"
OUTPUT_FILE = "data/training_set/time_ordered_training_data.parquet"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Target: 100 Million Rows
TARGET_ROWS = 100_000_000
LATENCY_THRESHOLD = 200.0


def generate_memory_safe_sample():
    con = duckdb.connect()
    # Set memory limit to 80% of your RAM to be safe (adjust as needed)
    con.execute("SET memory_limit='100GB'")

    print(f"--- Step 1: Scanning Dataset Statistics ---")
    start = time.time()

    # We need to know how many BAD vs GOOD rows exist to set the sample rate
    # This scan is fast because it's an aggregation, not a materialization
    stats_query = f"""
    SELECT 
        count(*) as total,
        SUM(CASE 
            WHEN packet_error_count > 0 OR rtt_max > {LATENCY_THRESHOLD} OR dup > 0 
            THEN 1 ELSE 0 
        END) as bad_count
    FROM '{INPUT_GLOB}'
    """
    print("Counting rows... (This might take a few minutes)")
    stats = con.execute(stats_query).fetchone()
    total_rows = stats[0]
    bad_rows = stats[1]
    good_rows = total_rows - bad_rows

    scan_time = time.time() - start
    print(f"Scan complete in {scan_time:.2f}s")
    print(f"Total: {total_rows:,} | Bad (Tail): {bad_rows:,} | Good: {good_rows:,}")

    # --- Step 2: Calculate Bernoulli Probabilities ---
    # Goal: Get max 100M rows. Prioritize BAD rows.

    if bad_rows >= TARGET_ROWS:
        # Scenario A: We have so much Bad data, we fill the whole buffer with it.
        p_bad = TARGET_ROWS / bad_rows
        p_good = 0.0
        print(f"Sampling Strategy: HUGE amount of tail data found.")
        print(f"Keeping {p_bad*100:.2f}% of Bad data. Dropping all Good data.")

    else:
        # Scenario B: Take all Bad data, fill the rest with Good.
        remaining_slots = TARGET_ROWS - bad_rows
        p_bad = 1.0  # Take all bad data
        p_good = remaining_slots / good_rows if good_rows > 0 else 0
        print(f"Sampling Strategy: Mixed Sample.")
        print(f"Keeping 100% of Bad data ({bad_rows:,}).")
        print(
            f"Filling remaining {remaining_slots:,} slots with {p_good*100:.4f}% of Good data."
        )

    # --- Step 3: Stream, Filter, Sort, Write ---
    print(f"\n--- Step 3: Generating Parquet File ---")

    # This query pushes the random() filter down to the scan.
    # It statistically picks the right number of rows WITHOUT sorting randomly.
    # The ORDER BY event_time happens only on the final ~100M result.
    final_query = f"""
    COPY (
        SELECT * EXCLUDE (is_bad)
        FROM (
            SELECT *,
                CASE 
                    WHEN packet_error_count > 0 OR rtt_max > {LATENCY_THRESHOLD} OR dup > 0 
                    THEN 1 ELSE 0 
                END as is_bad
            FROM '{INPUT_GLOB}'
        )
        WHERE 
            (is_bad = 1 AND random() < {p_bad})
            OR
            (is_bad = 0 AND random() < {p_good})
            
        -- Final Sort: Only sorts the ~100M result set, which fits in RAM.
        ORDER BY event_time ASC
    ) 
    TO '{OUTPUT_FILE}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
    """

    try:
        write_start = time.time()
        con.execute(final_query)
        write_elapsed = time.time() - write_start
        print(f"Success! Generation took {write_elapsed:.2f}s")
        print(f"Output saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    generate_memory_safe_sample()
