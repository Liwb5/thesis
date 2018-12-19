# coding:utf-8
import sys
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss,  optimizer, resume, config,
                 data_loader, metrics=None, valid_data_loader=None, lr_scheduler=None, train_logger=None, vocab=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.vocab = vocab
        self.do_validation = self.valid_data_loader is not None
        self.do_metrics = metrics is not None 
        self.lr_scheduler = lr_scheduler
        self.max_norm = config['trainer']['max_norm']
        #  self.log_step = int(np.sqrt(data_loader.batch_size))

    #  def _eval_metrics(self, output, target):
    #      acc_metrics = np.zeros(len(self.metrics))
    #      for i, metric in enumerate(self.metrics):
    #          acc_metrics[i] += metric(output, target)
    #          self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
    #      return acc_metrics

    def _compute_loss(self, predicts, labels):
        """ @predicts:(B, seq_len, vocab_size) 
            @labels: (B, seq_len). 
        """
        logits = torch.cat(predicts, 0)#(batch*seq_len, vocab_size)
        #logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.transpose(0,1).contiguous().view(-1)
        #  labels = labels.contiguous().view(-1)

        loss = torch.mean(self.loss(logits, labels))

        return loss


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        step_in_epoch = 0
        total_loss = 0
        for step, dataset in enumerate(self.data_loader):
            self.global_step += 1
            step_in_epoch += 1
            features, target, sents_len = self.vocab.summary_to_features(dataset['summaries'])
            features, target, sents_len  = Variable(features), Variable(target), Variable(sents_len)
            if self.device is not None:
                features = features.cuda()
                target = target.cuda()
                sents_len = sents_len.cuda()

            self.optimizer.zero_grad()
            #  tfr = self._update_teacher_forcing_ratio(self.global_step)
            tfr = 0.9
            probs, predicts = self.model(features, target, sents_len, tfr)
            loss = self._compute_loss(probs, target[:,1:])
            loss.backward()
            if self.max_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            #  self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            #  self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            if self.global_step % self.config['trainer']['print_every'] == 0:
                avg_loss = total_loss/self.config['trainer']['print_every']
                self.logger.info('Epoch: %d, global_batch: %d, Batch ID:%d Loss:%f'
                        %(epoch, self.global_step, step_in_epoch, avg_loss))
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                total_loss = 0
        log = {}
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        raise NotImplementedError
        #  self.model.eval()
        #  total_val_loss = 0
        #  total_val_metrics = np.zeros(len(self.metrics))
        #  with torch.no_grad():
        #      for batch_idx, (data, target) in enumerate(self.valid_data_loader):
        #          data, target = data.to(self.device), target.to(self.device)
        #
        #          output = self.model(data)
        #          loss = self.loss(output, target)
        #
        #          self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
        #          self.writer.add_scalar('loss', loss.item())
        #          total_val_loss += loss.item()
        #          total_val_metrics += self._eval_metrics(output, target)
        #          self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        #
        #  return {
        #      'val_loss': total_val_loss / len(self.valid_data_loader),
        #      'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        #  }
