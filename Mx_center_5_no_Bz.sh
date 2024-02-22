#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=M_x_center_5_no_Bz
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=10GB
module load anaconda3/2022.05
OUTDIR=$HOME/archive/2024/${SLURM_JOB_ID}
LOG=time_out/time_$(date +%F_%s_%N).txt
mkdir -p "${OUTDIR}"

# Save environment and job information to local directory.
local_dir=./slurm_job_info/${SLURM_JOB_ID}
mkdir -p "${local_dir}"
cp "$0" "${local_dir}"
env > "${local_dir}/env.txt"
echo "$0" > "${local_dir}/info.txt"
echo "$*" >> "${local_dir}/info.txt"

/usr/bin/time --output=${LOG} --verbose \
python3 monte_carlo_odmr_floquet_M_x.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}" \
  --param-start=1*pi*2e6 \
  --param-stop=0*pi*2e6 \
  --param-steps=51 \
  --mu-Mx=5*pi*2e6 \
  --Bx=0.0 \
  --By=0.0 \
  --Bz=0.0 \
  --omega-rf-power=0.0 \
  --omega-rf=0.0 \
  --n-avg=300
