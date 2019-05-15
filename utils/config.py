import yaml
import json
import datetime
import os
import sys
import shutil
from utils.util import make_dir

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def save_config(sourcefile, targetpath):
    filename = sourcefile.split('/')[-1]
    shutil.copyfile(sourcefile, targetpath+filename)

def get_config_from_yaml(yaml_file):
    """
    Get the config hyperparameters from yaml file.
    Using AttrDict comtainer to place these hyperparameters.
    """
    with open(yaml_file, 'r') as f:
        config = yaml.load(f)
    #  config = process_config(config)
    #  save_config(yaml_file, config['trainer']['args']['save_dir'])
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
    return config

#  def replace_config(new_config, old_config):
#      replaceable_params = ['epochs', 'n_gpu', 'batch_size', 'print_every', 'report_every']

def process_config(config):
    # make some necessary directories to save some important things
    time_stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    config['trainer']['args']['log_dir'] = ''.join((config['trainer']['args']['log_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp)))
    config['trainer']['args']['save_dir'] = ''.join((config['trainer']['args']['save_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp))) 
    config['trainer']['args']['output_dir'] = ''.join((config['trainer']['args']['output_dir'], config['task_name'], '/')) # , '.%s/' % (time_stamp)))
    make_dir(config['trainer']['args']['log_dir'])
    make_dir(config['trainer']['args']['save_dir'])
    make_dir(config['trainer']['args']['output_dir'])
    return config
