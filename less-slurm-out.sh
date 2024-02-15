#! /usr/bin/env sh
# tail --lines=+1 slurm-*.out | less
tail --lines=+1 $(find . -maxdepth 1 -name 'slurm-*.out' -print | sort -r) | less -c
