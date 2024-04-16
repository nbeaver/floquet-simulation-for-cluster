#! /bin/bash

for i in {1..20}
do 
  sbatch sbatch_with_RF_D_GS_center_2p87_with_Mx.sh
  sbatch sbatch_D_GS_center2p87_with_Mx.sh
done
