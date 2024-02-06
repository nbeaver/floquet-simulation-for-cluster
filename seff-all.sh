#! /usr/bin/env bash
prefix="slurm_"
for d in slurm_*;
do
    jobid=${d#"$prefix"}
    if ! test -f "${d}/seff_${jobid}"
    then
        seff "${jobid}" > "${d}/seff_${jobid}"
    fi
done
