#!/bin/bash
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
L=("3 nd_w" "3 w" "2 nd_w" "1 nd_w" "7 nd_f" "8 w_v" "5 nd_w" "6 nd_u" "4 nd_w" "9 nd_w" "1 nd_u" "2 nd" "6 nd_v" "7 w_f" "8 v" "4 nd_u" "5 nd" "9 nd_f" "3 nd" "2 w" "6 w_u" "7 nd_w" "8 nd_v" "9 w_f")
i=0
for j in "${L[@]}"; do set -- $j
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scratch/p2_lex.py $1 $2 400 > cache/p2_lex_r$1_$2.log 2>&1 &
  i=$((i+1)); if [ $((i % 6)) -eq 0 ]; then wait; fi
done
wait; echo ALLDONE > cache/p2_lex_done.txt
