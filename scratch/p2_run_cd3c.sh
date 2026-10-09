#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
until [ -f cache/p2_cd3b_done.txt ]; do sleep 15; done
L=()
for r in 8 3 2 6; do for k in nd_v_f nd_v_u nd_v_w v; do L+=("$r $k"); done; done
for r in 1 4 5 7; do for k in nd_v nd_v_f nd_v_u; do L+=("$r $k"); done; done
i=0
for j in "${L[@]}"; do set -- $j
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3_r$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait; echo ALLDONE > cache/p2_cd3c_done.txt
