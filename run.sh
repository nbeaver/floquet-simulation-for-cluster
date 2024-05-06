#! /bin/bash

for i in {1..80}
do 
    sbatch sbatch_Bz_center_0_with_Mx.sh
    sbatch sbatch_with_RF_Bz_center_0_with_Mx.sh
done
