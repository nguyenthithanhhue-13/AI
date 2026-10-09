#!/bin/bash
# chạy tuần tự theo lô 6 tiến trình
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
jobs_list=()
for k in nd nd_w nd_u nd_f; do for r in 1 2 3 4 5 6 7 8; do jobs_list+=("$r $k"); done; done
i=0
for j in "${jobs_list[@]}"; do
  set -- $j
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3_r$1_$2.log 2>&1 &
  i=$((i+1))
  if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait
echo ALLDONE > cache/p2_cd3_done.txt
