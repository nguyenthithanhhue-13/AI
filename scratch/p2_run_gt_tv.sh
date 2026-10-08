#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
for j in "2 gt_nd" "3 gt_w" "7 gt_f"; do set -- $j
  FITVAL=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3tv_r$1_$2.log 2>&1 &
done
wait; echo ALLDONE > cache/p2_gttv_done.txt
