#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
L=("2 tt_nd" "3 tt_w" "7 tt_f" "8 tt_v" "6 tt_u" "1 tt_nd" "4 tt_u" "5 tt_nd" "1 tt" "4 tt" "6 tt" "8 tt" "2 tt" "3 tt" "7 tt" "5 tt" "6 tt_v" "8 tt_nd")
i=0
for j in "${L[@]}"; do set -- $j
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3_r$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait; echo ALLDONE > cache/p2_tt_done.txt
