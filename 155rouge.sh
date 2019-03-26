#!/bin/sh
task_name="$1"
epoch="$2"
limit="$3"
if [ "$limit" = "b" ]; then
    echo "limit"
    python my_rouge.py -b_limit -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_${epoch}_275.txt 2>&1 &
else
    echo "no limit"
    python my_rouge.py -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_$epoch.txt 2>&1 &
fi
# source activate pytorch
