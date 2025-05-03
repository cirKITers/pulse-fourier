#!/bin/bash
#SBATCH --job-name=circuit_c12
#SBATCH --partition=cpuil
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB



# Execute the Python script
python scripts/circuit_15_run.py      # Assuming 'c12' corresponds to '_15'