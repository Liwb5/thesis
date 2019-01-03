# coding:utf-8
import os
import pickle
import datetime
import logging
import json
import argparse
import sys
import random 
import numpy as np

import torch
from torch.utils.data import DataLoader
import models
import models.loss as module_loss
import models.metrics as module_metrics
from trainer import Trainer
from utils import Logger
from utils.config import *
from data_loader import Vocab
from data_loader import Dataset

#  def get_instance(module, name, config, *args):
#      return getattr(module, config[name]['type'])(*args, **config[name]['args'])

#  def get_instance2(module, name, config):
#      return getattr(module, config[name]['type'])(config[name]['args'])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(config, resume):
    set_seed(config['seed'])

    log_format='%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s'
    logging.basicConfig(filename = ''.join((config['trainer']['log_dir'], 'log')),
                        filemode = 'w',
                        level = getattr(logging, config['log_level'].upper()),
                        format = log_format)

    logging.info(['config: ', config])
    train_logger = Logger()

    # setup data_loader instances
    train_data = Dataset(config['data_loader']['train_data'], 
                        data_quota = config['data_loader']['data_quota'])
    logging.info('using %d examples to train. ' % len(train_data))
    data_loader = DataLoader(dataset = train_data,
                            batch_size = config['data_loader']['batch_size'])

    val_data = Dataset(config['data_loader']['val_data'], 
                        data_quota=config['data_loader']['val_data_quota'])
    logging.info('using %d examples to val. ' % len(val_data))
    valid_data_loader = None  # TODO DEBUG
    #  valid_data_loader = DataLoader(dataset = val_data,
    #                          batch_size = config['data_loader']['batch_size'])

    vocab = Vocab(**config['vocabulary'], embed=None)

    # build model architecture
    #  model = get_instance2(models, 'model', config)
    model = getattr(models, config['model']['type'])(config['model']['args'], device=config['device'])
    logging.info(['model infomation: ', model])

    # get function handles of loss and metrics
    weights = torch.ones(config['model']['args']['vocab_size'])
    weights[vocab.PAD_ID] = 0
    loss = getattr(module_loss, config['loss'])(weights)

    metrics = getattr(module_metrics, config['metrics']) 


    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **config['optimizer']['args'])
    #  lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
    lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(**config['lr_scheduler']['args'])

    trainer = Trainer(model, loss,  optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      metrics=metrics,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger,
                      vocab = vocab)

    logging.info('begin training. ')
    trainer.train()

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
    
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)
