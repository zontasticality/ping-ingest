import duckdb
import os
import glob
import time

# Configuration
INPUT_DIR = "data/bz2_ping"
OUTPUT_DIR = "data/parquet_ping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the explicit schema based on our previous findings
# This allows DuckDB to skip the expensive "type inference" scan
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
    "result": "STRUCT(x VARCHAR)[ ]",  # Only keeping enough structure to extract errors
}


def convert_to_parquet(db_con, input_path):
    filename = os.path.basename(input_path)
    # Extract date from filename if possible for partitioning
    # Assuming format: ping-YYYY-MM-DD...
    # If not consistent, we can skip partitioning or use file modification time
    try:
        # Simple string parsing assuming "ping-2025-11-11" format
        date_part = filename.split("ping-")[1][:10]
    except:
        date_part = "misc"

    output_path = os.path.join(
        OUTPUT_DIR, f"{filename.replace('.json.bz2', '')}.parquet"
    )

    print(f"Processing {filename} -> {output_path}...")
    start = time.time()

    query = f"""
    COPY (
        SELECT 
            -- 1. SORT KEYS
            msm_id,
            dst_addr,
            to_timestamp(timestamp) as event_time,
            
            -- 2. METRICS
            min as rtt_min,
            avg as rtt_avg,
            max as rtt_max,
            sent,
            rcvd,
            dup,
            size,
            ttl,
            
            -- 3. ERROR EXTRACTION
            -- (DuckDB list lambda)
            len(list_filter(
                list_transform(result, item -> item.x), 
                x -> x IS NOT NULL
            )) as packet_error_count,

            -- 4. METADATA
            prb_id,
            src_addr,
            "from" as from_addr,
            dst_name,
            proto,
            af as ip_version,
            fw as firmware,
            mver as version,
            step,
            lts,
            group_id,
            
            -- 5. PARTITION COLUMN (Optional but recommended)
            '{date_part}' as capture_date

        FROM read_json('{input_path}', 
            columns={SCHEMA_DEF}, 
            format='newline_delimited'
        )
        
        -- SORTING FOR COMPRESSION & CLUSTERING
        ORDER BY msm_id, dst_addr, event_time
    ) 
    TO '{output_path}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
    """

    try:
        db_con.execute(query)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")
    except Exception as e:
        print(f"FAILED {filename}: {e}")


# Main execution loop
if __name__ == "__main__":
    # Use an on-disk DB for the conversion process to handle spillover if needed
    # though strictly speaking, COPY uses streaming so memory shouldn't spike.
    con = duckdb.connect("converter.db")

    # Get list of bz2 files
    print(os.listdir("data"))
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "ping-*")))

    print(f"Found {len(files)} files.")

    for f in files:
        convert_to_parquet(con, f)

    con.close()
    # cleanup temp db
    if os.path.exists("converter.db"):
        os.remove("converter.db")
