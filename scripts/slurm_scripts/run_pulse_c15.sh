#!/bin/bash
#SBATCH --job-name=pulse_c15_parallel
#SBATCH --partition=cpu
#SBATCH --time=07:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=170000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate

python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_15.py

