#!/bin/bash
#SBATCH --job-name=pulse_c12
#SBATCH --partition=htc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB


# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

# Execute the Python script
python ../pulse_15_run.py      # Assuming 'c12' corresponds to '_15' in your filenames