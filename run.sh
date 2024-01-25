#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=M_x_odmr_floquet
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=4100MB
module load anaconda3/2022.05
OUTDIR=/scratch/n.beaver/2024/${SLURM_JOB_ID}
LOG=time_$(date +%F_%s_%N).txt
mkdir -p "${OUTDIR}"
/usr/bin/time --output=${LOG} --verbose \
python3 monte_carlo_odmr_floquet_M_x.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}"
