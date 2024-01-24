#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=odmr_floquet
#SBATCH --partition=short
module load anaconda3/2022.05
LOG=time_$(date +%F_%s_%N).txt
/usr/bin/time --output=${LOG} --verbose \
python3 monte_carlo_odmr_floquet_B_x.py --verbose \
  --out-dir='/scratch/n.beaver/2024-01-23'
