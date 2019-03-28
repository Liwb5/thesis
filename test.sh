task_name=$1
epoch=$2
max_selected=$3
device=$4
limit=$5

fun() {
    python test.py -d $device -r ./checkpoints/$task_name/checkpoint-model_RL_AE-epoch$epoch.pth -m $max_selected &
    wait

    if [ $? -ne 0 ]; then
        echo "[ERROR] fail to run test.py."
        exit 1
    else
        echo "[INFO] successful to run test.py."
    fi

    # source activate root
    # wait

    if [ "$limit" = "b" ]; then
        echo "limit"
        python my_rouge.py -b_limit -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_${epoch}_${max_selected}_275.txt 2>&1 &
        wait
    else
        echo "no limit"
        python my_rouge.py -dir ./outputs/$task_name/ -epoch $epoch > ./outputs/$task_name/rouge155_result_${epoch}_${max_selected}.txt 2>&1 &
        wait
    fi

    if [ $? -ne 0 ]; then
        echo "[ERROR] fail to run my_rouge.py."
        exit 1
    else
        echo "[INFO] successful to run my_rouge.py."
    fi

    # source activate pytorch
}

fun
