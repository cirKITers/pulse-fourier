#!/bin/bash
#SBATCH --job-name=pulse_c9
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=32000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate

python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/pulse_9_run.py

