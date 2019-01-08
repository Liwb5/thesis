import yaml
import json
import datetime
import os
import sys
from utils.util import make_dir

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config_from_yaml(yaml_file):
    """
    Get the config hyperparameters from yaml file.
    Using AttrDict comtainer to place these hyperparameters.
    """
    with open(yaml_file, 'r') as f:
        config = yaml.load(f)
    config = process_config(config)
    return config 


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary) 
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config = json.load(config_file)
    config = process_config(config)
    return config

#  def replace_config(new_config, old_config):
#      replaceable_params = ['epochs', 'n_gpu', 'batch_size', 'print_every', 'report_every']

def process_config(config):
    # make some necessary directories to save some important things
    time_stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    config['trainer']['log_dir'] = ''.join((config['trainer']['log_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp)))
    config['trainer']['save_dir'] = ''.join((config['trainer']['save_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp))) 
    config['trainer']['output_dir'] = ''.join((config['trainer']['output_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp)))
    make_dir(config['trainer']['log_dir'])
    make_dir(config['trainer']['save_dir'])
    make_dir(config['trainer']['output_dir'])
    return config
