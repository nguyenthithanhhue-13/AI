#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
i=0
for j in "1 nd_u" "2 nd" "3 w" "4 nd_u" "5 nd" "6 nd_v_u" "7 nd_f" "8 v"; do set -- $j
  FITVAL=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_cd3.py $1 $2 > cache/p2_cd3tv_r$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait; echo ALLDONE > cache/p2_tv_done.txt
