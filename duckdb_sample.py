import duckdb
import pandas as pd

INPUT_FILE = "data/training_set/minimal_training_data.parquet"


def diagnose_and_sample():
    con = duckdb.connect()

    print("--- 1. DIAGNOSTIC: Checking Time Range ---")
    try:
        # Check if the file actually covers the 11th to the 25th
        stats_query = f"""
        SELECT 
            count(*) as total_rows,
            min(event_time) as start_time,
            max(event_time) as end_time,
            (max(event_time) - min(event_time)) as duration_interval
        FROM '{INPUT_FILE}'
        """
        stats = con.sql(stats_query).df()
        print(stats.to_string(index=False))

        # Check if we have the expected 14-day range
        duration = stats.iloc[0]["duration_interval"]
        if pd.isnull(duration) or duration.days < 1:
            print("\n[WARNING] Dataset duration is less than 1 day!")
            print("Check your aggregation script: did it detect both input files?")
        else:
            print(f"\n[OK] Dataset covers {duration.days} days.")

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\n--- 2. SAMPLING: True Global Random Sequences ---")

    # We create a probability that picks roughly 25 rows out of the total.
    # P = 25 / Total_Rows. We use slightly higher (50) to be safe and Limit.
    total_rows = stats.iloc[0]["total_rows"]
    if total_rows == 0:
        return

    sample_rate = 100.0 / total_rows  # Target ~100 rows, pick 25

    for i in range(1, 6):
        print(f"### Random Sequence {i} ###")

        # This query forces a full table scan, ensuring we pick rows from
        # the start (11th) AND the end (25th) uniformly.
        query = f"""
        SELECT * FROM (
            SELECT 
                event_time,
                msm_id,
                src_addr,
                dst_addr,
                rtt,
                packet_error_count as err
            FROM '{INPUT_FILE}'
            WHERE random() < {sample_rate} 
            LIMIT 25
        ) 
        ORDER BY event_time ASC
        """

        df = con.sql(query).df()

        # Clean formatting
        pd.set_option("display.max_colwidth", 40)
        pd.set_option("display.width", 1000)

        print(df.to_string(index=False))
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    diagnose_and_sample()
