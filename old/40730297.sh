#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=M_x_center_5_no_Bz
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=10GB
module load anaconda3/2022.05
OUTDIR=/scratch/n.beaver/2024/${SLURM_JOB_ID}
LOG=time_$(date +%F_%s_%N).txt
mkdir -p "${OUTDIR}"

# Save environemnt and job information to local directory.
local_dir=./slurm_${SLURM_JOB_ID}
mkdir -p "${local_dir}"
cp "$0" "${local_dir}"
env > "${local_dir}/env.txt"
echo "$0" > "${local_dir}/info.txt"
echo "$*" >> "${local_dir}/info.txt"

/usr/bin/time --output=${LOG} --verbose \
python3 monte_carlo_odmr_floquet_M_x.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}" \
  --param-stop=6.28e6 \
  --param-steps=50 \
  --n-avg=300
