#! /bin/bash

for i in {1..10}
do 
  sbatch Bz_center_0_with_Mx.sh
  sleep 1
done
