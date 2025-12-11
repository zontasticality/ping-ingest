import duckdb
import os
import pandas as pd

# Path to your generated training file
INPUT_FILE = "data/training_set/training_data.parquet"


def inspect_column_sizes():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    print(f"Inspecting metadata for: {INPUT_FILE}")
    con = duckdb.connect()

    # query explanation:
    # 1. parquet_metadata reads only the file footer (instant, no matter file size).
    # 2. We sum the chunk sizes for every row group to get total column weight.
    query = f"""
    WITH col_stats AS (
        SELECT 
            path_in_schema as column_name,
            SUM(total_compressed_size) as comp_bytes,
            SUM(total_uncompressed_size) as uncomp_bytes,
            -- Calculate compression ratio (Higher is better)
            CAST(SUM(total_uncompressed_size) AS DOUBLE) / NULLIF(SUM(total_compressed_size), 0) as ratio
        FROM parquet_metadata('{INPUT_FILE}')
        GROUP BY path_in_schema
    )
    SELECT 
        column_name,
        -- Convert to MB for readability
        round(comp_bytes / 1024.0 / 1024.0, 2) as compressed_mb,
        round(uncomp_bytes / 1024.0 / 1024.0, 2) as uncompressed_mb,
        round(ratio, 1) as compression_ratio,
        -- Calculate % of total file size
        round(comp_bytes * 100.0 / (SELECT SUM(comp_bytes) FROM col_stats), 1) as pct_of_disk
    FROM col_stats
    ORDER BY comp_bytes DESC
    """

    try:
        df = con.sql(query).df()

        # Calculate Totals
        total_comp = df["compressed_mb"].sum()
        total_uncomp = df["uncompressed_mb"].sum()

        print(
            f"\n--- Total File Size: {total_comp:.2f} MB (Uncompressed footprint: {total_uncomp:.2f} MB) ---\n"
        )
        print(df.to_string(index=False))

        print("\n--- Analysis & Tips ---")
        top_col = df.iloc[0]
        print(
            f"1. Heaviest Column: '{top_col['column_name']}' takes up {top_col['pct_of_disk']}% of the file."
        )

        low_ratio = df[df["compression_ratio"] < 2.0]
        if not low_ratio.empty:
            print(
                f"2. Poor Compression: These columns compress poorly (< 2x): {low_ratio['column_name'].tolist()}"
            )
            print(
                "   (Consider dictionary encoding or dropping them if they are high-cardinality strings)"
            )
        else:
            print("2. Compression looks healthy across all columns.")

    except Exception as e:
        print(f"Error reading metadata: {e}")
        print("Note: This script requires DuckDB v0.8.0 or newer.")


if __name__ == "__main__":
    inspect_column_sizes()
