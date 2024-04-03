#! /bin/bash

for i in {1..20}
do 
  sbatch sbatch_with_RF_Bx_center_0_with_Mx.sh
  sbatch sbatch_Bx_center_0_with_Mx.sh
done
