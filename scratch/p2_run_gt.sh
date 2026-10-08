#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
L=("1 gt" "4 gt" "5 gt" "6 gt" "7 gt" "8 gt" "1 gt_nd" "5 gt_nd" "6 gt_f" "7 gt_f" "8 gt_v" "4 gt_u" "1 gt_u" "6 gt_u" "4 gt_nd" "7 gt_nd" "8 gt_nd" "2 gt_f" "3 gt_f" "2 gt_u" "3 gt_v" "6 gt_v" "1 gt_w" "5 gt_w")
i=0
for j in "${L[@]}"; do set -- $j
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3_r$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait; echo ALLDONE > cache/p2_gt_done.txt
