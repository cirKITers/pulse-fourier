#!/bin/bash
#SBATCH --job-name=test_pulse_c9_parallel
#SBATCH --partition=cpu
#SBATCH --time=10:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=350000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate

python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_9.py

