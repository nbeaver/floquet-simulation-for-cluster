#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=test
#SBATCH --partition=short
module load anaconda3/2022.05
LOG=time_$(date +%F_%s_%N).txt
/usr/bin/time --output=${LOG} --verbose \
python3 odmr_floquet_simulation.py
