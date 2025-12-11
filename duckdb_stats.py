import duckdb
import os

INPUT_FILE = "data/training_set/training_data.parquet"


def check_dst_addr():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    con = duckdb.connect()

    print(f"--- Investigating 'dst_addr' in {INPUT_FILE} ---")

    # 1. Check Sparsity vs Cardinality
    query_stats = f"""
    SELECT 
        count(*) as total_rows,
        count(dst_addr) as non_null_rows,
        count(DISTINCT dst_addr) as unique_destinations,
        
        -- Percentage of rows that have a value
        round(count(dst_addr) * 100.0 / count(*), 2) as population_pct,
        
        -- Diversity: If this is low (< 1%), dictionary encoding is doing heavy lifting
        round(count(DISTINCT dst_addr) * 100.0 / count(*), 4) as distinct_pct
    FROM '{INPUT_FILE}'
    """

    stats = con.sql(query_stats).df()
    print(stats.to_string(index=False))

    # 2. See the most common values (The Dictionary)
    print("\n--- Top 10 Destinations (The Dictionary) ---")
    query_top = f"""
    SELECT 
        dst_addr, 
        count(*) as freq,
        round(count(*) * 100.0 / (SELECT count(*) FROM '{INPUT_FILE}'), 2) as pct_of_dataset
    FROM '{INPUT_FILE}'
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 10
    """
    print(con.sql(query_top).df().to_string(index=False))


if __name__ == "__main__":
    check_dst_addr()
