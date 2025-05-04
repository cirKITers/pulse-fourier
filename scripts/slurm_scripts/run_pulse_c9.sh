#!/bin/bash
#SBATCH --job-name=pulse_c9
#SBATCH --partition=htc
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB


python3 -m venv .venv

source .venv/bin/activate

pip install -e .

python scripts/test_run.py


# Execute the Python script
python ../pulse_9_run.py

