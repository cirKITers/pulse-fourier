#!/bin/bash
#SBATCH --job-name=pulse_c9
#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB


# Execute the Python script
python ../pulse_9_run.py