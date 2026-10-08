#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
for r in 3 6; do
  for cfg in "0.01 10 32" "0.05 15 32" "0.01 15 64"; do set -- $cfg
    WD=$1 VTAG=reg PYTHONPATH=F:/pylibs PYTHONUTF8=1 python scratch/p2_vin2.py $r $2 $3 3 >> cache/p2_vinreg_r$r.log 2>&1
  done
done
echo DONE >> cache/p2_vinreg_done.txt
