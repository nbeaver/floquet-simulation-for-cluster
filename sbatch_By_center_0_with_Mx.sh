#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=B_y_center_0_with_Mx
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=FAIL,INVALID_DEPEND
#SBATCH --mem-per-cpu=10GB
#SBATCH --output=slurm_job_info/%j/slurm-%j.out
module load anaconda3/2022.05
OUTDIR=/work/pstevenson/n.beaver/2024/floquet-simulations/${SLURM_JOB_ID}
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
python3 monte_carlo_odmr_floquet_B_y.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}" \
  --param-start=1.0e-4 \
  --param-stop=0.0e-4 \
  --param-steps=51 \
  --mu-By=0.0e-4 \
  --Mx='2*pi*5e6' \
  --Bz=0.0 \
  --Bx=0.0 \
  --omega-rf-power=0.0 \
  --omega-rf=0.0 \
  --n-avg=100
