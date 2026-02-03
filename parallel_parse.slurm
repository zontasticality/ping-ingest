#!/bin/bash
#SBATCH --job-name=ping-parse
#SBATCH --output=parse_%j.out
#SBATCH --error=parse_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=20

# ---------------------------------------------------------------
# Slurm batch wrapper for duckdb_parallel_parse.py
#
# Usage:
#   sbatch parallel_parse.slurm                     # defaults: all files, 8 workers, random order, local tmp
#   sbatch parallel_parse.slurm -n 10               # process 10 files
#   sbatch parallel_parse.slurm -w 4 --no-random    # 4 workers, sorted order
#   sbatch parallel_parse.slurm --dry-run           # preview which files would be processed
#
# All arguments after the sbatch script are forwarded to the
# Python script. See `python duckdb_parallel_parse.py --help`
# for the full list of options.
# ---------------------------------------------------------------

set -euo pipefail

PROJECT_DIR=/scratch4/workspace/zevwilson_umass_edu-pingdata/ping-ingest
cd "$PROJECT_DIR"

echo "=== Slurm Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Memory:    $SLURM_MEM_PER_NODE MB"
echo "Started:   $(date)"
echo "======================"

# Activate virtual environment
source .venv/bin/activate

# Disable Python output buffering so worker logs appear in real time
export PYTHONUNBUFFERED=1

# Run the parallel parser
# --local-tmp: uses node-local /tmp, avoids NFS bottleneck
# -w 8: 8 workers with 24GB each
python duckdb_parallel_parse.py --local-tmp -w 8 "$@"

echo ""
echo "=== Job finished: $(date) ==="
