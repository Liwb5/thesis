device=$1
task_name=$2
# python main.py -c ./configs/debug_config.yaml
# python main.py -c ./configs/test_config.yaml
python main.py -d $device -n $task_name -c ./configs/config.yaml
