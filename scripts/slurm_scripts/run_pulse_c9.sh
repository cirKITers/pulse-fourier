#!/bin/bash
#SBATCH --job-name=pulse_c9_parallel
#SBATCH --partition=cpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate


seeds=("10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29")

for seed in "${seeds[@]}"; do
  echo "Running with seed: $seed"
  python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_9.py "$seed"
done


