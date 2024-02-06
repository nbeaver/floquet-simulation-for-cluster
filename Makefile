run:
	sbatch run.sh

less-all-slurm-out:
	tail --lines=+1 slurm-*.out | less

less-all-time-out:
	tail --lines=+1 time_*.txt | less
