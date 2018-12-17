# coding:utf-8
import os
import pickle
import logging
import json
import argparse
import sys
import random 
import numpy as np

import torch
import data_loader.DataLoader as module_data
import models.loss as module_loss
#  import model.metric as module_metric
import models
#  from trainer import Trainer
#  sys.path.append('./data_loader')
#  from DataLoader import BatchDataLoader, PickleReader
from utils.logger import Logger
from utils.config import *
sys.path.append('./data_loader')
from data_loader.Vocab import Vocab
from data_loader.Dataset import Document, Dataset

#  def get_instance(module, name, config, *args):
    #  return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(config[name]['args'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config, resume):
    set_seed(config['seed'])

    log_format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s'
    logging.basicConfig(filename = config['trainer']['log_dir'],
                        filemode = 'w',
                        level = getattr(logging, config['log_level'].upper()),
                        format = log_format)

    logging.info(['config: ', config])
    logging.info('initializing logger. ')
    train_logger = Logger()

    # setup data_loader instances
    logging.info('initializing data_loader. ')
    data_loader = module_data.PickleReader(config['data_loader']['data_dir'])
    #  valid_data_loader = data_loader.split_validation()
    logging.info('initializing vocabulary. ')
    with open(config['data_loader']['vocab_file'], 'rb') as f:
        vocab = pickle.load(f)

    # build model architecture
    logging.info('initializing model. ')
    model = get_instance(models, 'model', config)
    logging.info(['model infomation: ', model])

    # get function handles of loss and metrics
    logging.info('initializing loss. ')
    weights = torch.ones(config['model']['args']['vocab_size'])
    weights[vocab.PAD_ID] = 0
    loss = getattr(module_loss, config['loss'])(weights)
    #  metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    #  trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #  optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    #  lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    #
    #  trainer = Trainer(model, loss, metrics, optimizer,
    #                    resume=resume,
    #                    config=config,
    #                    data_loader=data_loader,
    #                    valid_data_loader=valid_data_loader,
    #                    lr_scheduler=lr_scheduler,
    #                    train_logger=train_logger)
    #
    #  trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='thesis')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = get_config_from_json(args.config)
        #  path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
