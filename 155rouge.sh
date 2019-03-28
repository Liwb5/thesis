#!/bin/sh
task_name="$1"
epoch="$2"
max_selected=$3
limit="$4"

if [ "$limit" = "b" ]; then
    echo "limit"
    python my_rouge.py -b_limit -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_${epoch}_${max_selected}_275.txt 2>&1 &
else
    echo "no limit"
    python my_rouge.py -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_$epoch_${max_selected}.txt 2>&1 &
fi
# source activate pytorch
