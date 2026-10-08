#!/bin/bash
# huấn luyện mô hình siêu tuyến tính với ĐIỀU KIỆN RIÊNG từng robot (chọn theo độ nhạy, scratch/p2_hyper_sens.py), 5 hạt giống
cd /f/Research/Phenikaa_Campus_Courier_2026/AI
MODE=${1:-}
declare -A SEL=( [1]="rain+night+urg" [2]="rain+night+goal" [3]="rain+goal" [4]="rain+urg+frag+goal" [5]="night"
                 [6]="night+urg+frag+via+mapg+goal" [7]="rain+night+urg+frag+goal" [8]="night+urg+frag+via+goal" )
for r in 1 2 3 4 5 6 7 8; do
  COND=${SEL[$r]} HTAG=sel PYTHONPATH=F:/pylibs PYTHONUTF8=1 python scratch/p2_hyper.py $r 1500 64 5 $MODE 2>&1 | grep -E "hyper|Trace|Error" | sed "s/^/[${SEL[$r]}] /"
done
echo DONE
