#!/bin/bash
#SBATCH --job-name=pulse_hea_parallel_seeds_251_to_500
#SBATCH --partition=cpu
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate

start=251
end=500

for ((seed=start; seed<=end; seed++)); do
  echo "Running with seed: $seed"
  python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_hea.py "$seed"
done