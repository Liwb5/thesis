import yaml
import json
import os
import sys

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_config_from_yaml(yaml_file):
    """
    Get the config hyperparameters from yaml file.
    Using AttrDict comtainer to place these hyperparameters.
    """
    return AttrDict(yaml.load(open(yaml_file, 'r')))


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

#  def process_config(config):
#      config['trainer']['log_dir'] = os.path.join(config['trainer']['log_dir'], config['name'], '.log')
#      config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
#      return config
