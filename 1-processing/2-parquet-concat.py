import glob
import os
import dask
from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd

# It's good practice to start the client at the top
client = Client(processes=False, memory_limit='60GB')
print(f"Dask Dashboard: {client.dashboard_link}")

# --- Stage 1: File Consolidation (for Nested Directories) ---

# 1. Define source and destination
source_root = "data/ping/"
target_path = "data/ping_consolidated/"
os.makedirs(target_path, exist_ok=True)

# 2. Get a list of all your individual data files using a recursive glob
# The pattern '**/part*.parquet' will search through all subdirectories
# for files that start with 'part' and end with '.parquet'.
glob_pattern = os.path.join(source_root, '**', 'part*.parquet')
print(f"Searching for files with pattern: {glob_pattern}")

all_files = sorted(glob.glob(glob_pattern, recursive=True))

if not all_files:
    raise FileNotFoundError(f"No files found matching the pattern '{glob_pattern}'. Please check your source_root path.")

print(f"Found {len(all_files)} individual part files across all subdirectories.")

# 3. Group files into manageable chunks
# With ~98k files, let's aim for ~500 output files. 98000 / 500 = 196.
# A chunksize of 200 is a good starting point. Adjust if needed.
chunksize = 200
file_chunks = [all_files[i:i + chunksize] for i in range(0, len(all_files), chunksize)]
print(f"Grouped into {len(file_chunks)} chunks of up to {chunksize} files each.")

# 4. Create a Dask Delayed task for each chunk
@dask.delayed
def consolidate_chunk(chunk_of_files, output_filename):
    """Reads a list of parquet files and writes them as a single new file."""
    # This reads a small, manageable list of files into a Dask DataFrame
    df_chunk = dd.read_parquet(chunk_of_files, engine="pyarrow")
    
    # This triggers the read and writes the concatenated result to a single Parquet file.
    df_chunk.to_parquet(
        output_filename,
        engine="pyarrow",
        write_metadata_file=False, # We'll create a single one at the end
        compression="snappy"
        # We can often omit the schema here; to_parquet is good at inferring it.
        # If you get errors, you can add it back:
        # schema=df_chunk.head(1).iloc[0:0].to_arrow_schema()
    )
    return output_filename

# 5. Build and run the consolidation tasks
tasks = []
for i, chunk in enumerate(file_chunks):
    # Create a flat output structure
    output_file = os.path.join(target_path, f"part-{i:04d}.parquet")
    tasks.append(consolidate_chunk(chunk, output_file))

print("Submitting consolidation tasks to Dask...")
# This will run all the consolidation tasks in parallel.
# Each task is independent, so the graph is very simple and efficient.
dask.compute(*tasks)

# 6. Optional but highly recommended: Write a central _metadata file for the new dataset
print("Creating a central _metadata file for the consolidated dataset...")
dd.io.parquet.create_metadata_file(
    [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.parquet')],
    out_dir=target_path
)

print(f"Consolidation complete. Your new, efficient dataset is ready in {target_path}")
client.close()