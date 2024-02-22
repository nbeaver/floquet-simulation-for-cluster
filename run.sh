#! /bin/bash

for i in {1..10}
do 
  sbatch sbatch_Bz_center_0_with_Mx.sh
  sbatch sbatch_Bz_center_5_no_Mx.sh
  sbatch sbatch_Bz_center_5_with_Mx.sh
  sbatch sbatch_Mx_center_0_with_Bz.sh
  sbatch sbatch_Mx_center_2_no_Bz.sh
  sbatch sbatch_Mx_center_2_with_Bz.sh
  sbatch sbatch_Mx_center_5_no_Bz.sh
  sleep 1
done
