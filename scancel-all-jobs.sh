#! /usr/bin/env sh
squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs -n 1 scancel
