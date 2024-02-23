#! /usr/bin/env bash
prefix="slurm_job_info/"
for d in slurm_job_info/*;
do
    jobid=${d#"$prefix"}
    seff "${jobid}" > "${d}/seff_${jobid}"
done
