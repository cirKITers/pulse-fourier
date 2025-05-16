#!/bin/bash
#SBATCH --job-name=pulse_c9_parallel
#SBATCH --partition=cpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320000MB
#SBATCH --output="logs/slurm-%j-%x.out"

module load compiler/llvm

module load devel/python/3.11.7

source /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/.venv/bin/activate


seeds=("30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40" "41" "42" "43" "44" "45" "46" "47" "48" "49" "50" "51" "52" "53" "54" "55" "56" "57" "58" "59" "60" "61" "62" "63" "64" "65" "66" "67" "68" "69" "70" "71" "72" "73" "74" "75" "76" "77" "78" "79" "80" "81" "82" "83" "84" "85" "86" "87" "88" "89" "90" "91" "92" "93" "94" "95" "96" "97" "98" "99" "100" "101" "102" "103" "104" "105" "106" "107" "108" "109")

for seed in "${seeds[@]}"; do
  echo "Running with seed: $seed"
  python /pfs/data6/home/ka/ka_scc/ka_tc6850/pulse-fourier/scripts/parallel_pulse_9.py "$seed"
done


