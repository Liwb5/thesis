task_name=$1
# python main.py -c ./configs/debug_config.yaml
# python main.py -c ./configs/test_config.yaml
python main.py -c ./configs/config.yaml -n $task_name
