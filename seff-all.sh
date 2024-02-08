#! /usr/bin/env bash
prefix="slurm_"
for d in slurm_*;
do
    jobid=${d#"$prefix"}
    seff "${jobid}" > "${d}/seff_${jobid}"
done
