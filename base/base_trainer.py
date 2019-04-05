# coding:utf-8
import os
import math
import json
import logging
import datetime
import torch
from utils.util import * 
#  from utils.visualization import WriterTensorboardX
from tensorboardX import SummaryWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device = config['device']
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.exp_avg_reward = torch.zeros(1)

        if self.device is not None:
            #  torch.cuda.set_device(config['device'])
            self.model = model.cuda()
            self.loss = loss.cuda()
            self.exp_avg_reward = self.exp_avg_reward.cuda()


        self.train_logger = train_logger

        self.use_summaryWriter = config['use_summaryWriter']
        self.batch_size = self.config['data_loader']['batch_size']
        self.trainer_config = config['trainer']['args']
        self.epochs = self.trainer_config['epochs']
        self.save_period = self.trainer_config['save_period']
        self.start_epoch = 1
        self.global_step = 0
        self.reward_type = config['trainer']['reward_type']

        # setup visualization writer instance
        #  writer_save_path = ''.join((self.trainer_config['log_dir'], 'tensorboardx.log'))
        if self.use_summaryWriter:
            self.writer = SummaryWriter(self.trainer_config['log_dir']) # tensorboard 建立的是目录，它会自动产生文件名，不需要手动指定

        #  Save configuration file into checkpoint directory:
        #  config_save_path = os.path.join(self.trainer_config['save_dir'], 'config.json')
        #  with open(config_save_path, 'w') as handle:
        #      json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    
    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_result = {}
            train_result = self._train_epoch(epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            val_result = {}
            if self.trainer_config['do_validation'] and self.do_validation:
                self.logger.info('doing validation ... ')
                val_result = self._valid_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch, **train_result, **val_result}
            if self.train_logger is not None:
                self.train_logger.add_entry(log) # record some information so that we can save it in a checkpoint

            self.logger.info(log)
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=None)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_name = type(self.model).__name__
        state = {
            'model': model_name,
            'epoch': epoch,
            'global_step': self.global_step, 
            #  'exp_avg_reward': self.exp_avg_reward,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            #  'state_dict': self.model.cpu().state_dict(), #using cpu() to save state_dict into cpu mode, better
            'optimizer': self.optimizer.state_dict(),
            #  'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.trainer_config['save_dir'], 'checkpoint-model_{}-epoch{}.pth'.format(model_name, epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc:storage) # load parameters to CPU
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step'] + 1
        #  self.exp_avg_reward = checkpoint['exp_avg_reward']
        #  self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.device is not None:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
    
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
