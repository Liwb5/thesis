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
        if self.device is not None:
            torch.cuda.set_device(config['device'])
            self.model = model.cuda()
            self.loss = loss.cuda()

        self.metrics = metrics
        self.optimizer = optimizer
        self.train_logger = train_logger

        self.cfg_trainer = config['trainer']
        self.epochs = self.cfg_trainer['epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.print_every = self.cfg_trainer['print_every']
        self.start_epoch = 1
        self.global_step = 0

        # setup visualization writer instance
        #  writer_save_path = ''.join((self.cfg_trainer['log_dir'], 'tensorboardx.log'))
        self.writer = SummaryWriter(self.cfg_trainer['log_dir']) # tensorboard 建立的是目录，它会自动产生文件名，不需要手动指定

        #  Save configuration file into checkpoint directory:
        config_save_path = os.path.join(self.cfg_trainer['save_dir'], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
    
    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            
            # save logged informations into log dict
            log = {'epoch': epoch}
            # evaluate model performance according to configured metric, save best checkpoint as model_best
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
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            #  'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.config['trainer']['save_dir'], 'checkpoint-model_{}-epoch{}.pth'.format(model_name, epoch))
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
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))