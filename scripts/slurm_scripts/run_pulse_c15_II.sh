#!/bin/bash
#SBATCH --job-name=pulse_c15_parallel_seed_51_to_2275
#SBATCH --partition=cpu
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate


start=51
end=275   # inclusive

for ((seed=start; seed<=end; seed++)); do
  echo "Running with seed: $seed"
  python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_15.py "$seed"
done
