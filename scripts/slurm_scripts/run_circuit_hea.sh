#!/bin/bash
#SBATCH --job-name=circuit_hea
#SBATCH --partition=cpuil
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB


# Execute the Python script
python scripts/circuit_hea_run.py