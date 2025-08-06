#!/bin/bash
#SBATCH --job-name=ping-optimize
#SBATCH --output=optimize_%j.out
#SBATCH --error=optimize_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=48

cd /scratch/workspace/zevwilson_umass_edu-simple/ping-ingest
python 4-optimize-dataset.py