import duckdb
import os
import glob
import time

INPUT_DIR = "data/bz2_ping"
OUTPUT_DIR = "data/parquet_ping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Explicit Schema (Faster & Safer)
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


def convert_to_parquet(db_con, input_path):
    filename = os.path.basename(input_path)
    output_path = os.path.join(
        OUTPUT_DIR, f"{filename.replace('.json.bz2', '')}.parquet"
    )

    print(f"Processing {filename}...")
    start = time.time()

    query = f"""
    COPY (
        SELECT 
            -- 1. TIME (Primary Sort Key now)
            to_timestamp(timestamp) as event_time,
            
            -- 2. KEYS
            msm_id, dst_addr, prb_id,
            
            -- 3. METRICS
            min as rtt_min, avg as rtt_avg, max as rtt_max,
            sent, rcvd, dup, size, ttl,
            
            -- 4. ERRORS (Connection Tails)
            len(list_filter(list_transform(result, item -> item.x), x -> x IS NOT NULL)) as packet_error_count,

            -- 5. METADATA
            src_addr, "from" as from_addr, dst_name, proto, af as ip_version,
            fw as firmware, mver as version, step, lts, group_id

        FROM read_json('{input_path}', columns={SCHEMA_DEF}, format='newline_delimited')
        
        -- CHANGED: Sort by Time first.
        -- This ensures the parquet file is optimized for temporal queries.
        ORDER BY event_time ASC, msm_id
    ) 
    TO '{output_path}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD', ROW_GROUP_SIZE 100000);
    """

    try:
        db_con.execute(query)
        print(f"  -> Done in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    # Ingestion doesn't need huge memory, it streams.
    con = duckdb.connect()
    print(os.listdir("data/bz2_ping"))
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "ping-*")))
    print(files)
    for f in files:
        convert_to_parquet(con, f)
    con.close()
