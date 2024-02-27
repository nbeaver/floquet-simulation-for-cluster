#! /usr/bin/env sh
#tail --lines=+1 slurm_job_info/*/seff_* | less
tail --lines=+1 $(find slurm_job_info/ -maxdepth 2 -name 'seff_*' -print | sort -r) | less -c
