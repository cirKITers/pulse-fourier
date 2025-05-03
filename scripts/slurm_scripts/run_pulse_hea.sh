#!/bin/bash
#SBATCH --job-name=pulse_hea
#SBATCH --partition=htc
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB



# Execute the Python script
python scripts/pulse_hea_run.py