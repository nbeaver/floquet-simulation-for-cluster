#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=B_z_odmr_floquet
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=ALL
module load anaconda3/2022.05
OUTDIR=/scratch/n.beaver/2024-01-24
LOG=time_$(date +%F_%s_%N).txt
mkdir -p "${OUTDIR}"
/usr/bin/time --output=${LOG} --verbose \
python3 monte_carlo_odmr_floquet_B_z.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}"
