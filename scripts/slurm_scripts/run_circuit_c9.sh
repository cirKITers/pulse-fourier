#!/bin/bash
#SBATCH --job-name=circuit_c9
#SBATCH --partition=cpuil
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB



# Execute the Python script
python scripts/circuit_9_run.py