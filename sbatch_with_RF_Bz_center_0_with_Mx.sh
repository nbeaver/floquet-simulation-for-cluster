#! /bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=3-16_with_RF_B_z_center_0_with_Mx
#SBATCH --partition=short
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=7GB
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
python3 monte_carlo_odmr_floquet_B_z.py --verbose \
  --out-dir="${OUTDIR}" \
  --tag="${SLURM_JOB_ID}" \
  --param-start=16.0e-4 \
  --param-stop=3.0e-4 \
  --param-steps=14 \
  --mu-Bz=0e-4 \
  --Mx='2*pi*5e6' \
  --Bx=0.0 \
  --By=0.0 \
  --omega-rf-power=2*pi*2e6 \
  --omega-rf=10*pi*2e6 \
  --MW-start-freq=2.855*GHz \
  --MW-stop-freq=2.885*GHz \
  --MW-step=100*kHz \
  --n-avg=1000
