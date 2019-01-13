# coding:utf-8
import sys
import logging
import math
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from pprint import pprint, pformat


class AE_trainer(BaseTrainer):
    """
    AE_trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss,  optimizer, resume, config,
                 data_loader, metrics=None, valid_data_loader=None, lr_scheduler=None, train_logger=None, vocab=None):
        super(AE_trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.vocab = vocab
        self.do_validation = self.valid_data_loader is not None
        self.do_metrics = metrics is not None 
        self.lr_scheduler = lr_scheduler
        self.max_norm = self.trainer_config['max_norm']
        #  self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, predicts, reference):
        hypothesis = self.vocab.features_to_tokens(predicts.numpy().tolist())
        results = self.metrics(hypothesis, reference)
        return results
        
    #  def predict(self, predicts, target):
    #      """
    #      @ target: Varialbe, (B, L)
    #      """
    #      predicted_tokens = self.vocab.features_to_tokens(predicts.numpy().tolist())
    #      target_tokens = self.vocab.features_to_tokens(target.data.cpu().numpy().tolist())
    #      self.logger.info(['hyp: ', predicted_tokens])
    #      self.logger.info(['ref: ', target_tokens])

    def _update_tfr(self, epoch):
        tfr = max((1.0-0.1*epoch), 0.5)
        #  tfr = self.trainer_config['teacher_forcing_ratio'] - self.global_step/len(self.data_loader)
        return tfr 

    def _compute_loss(self, predicts, labels):
        """ @predicts:(B, seq_len, vocab_size) 
            @labels: (B, seq_len). 
        """
        logits = torch.cat(predicts, 0)#(batch*seq_len, vocab_size)
        #logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.transpose(0,1).contiguous().view(-1)
        #  labels = labels.contiguous().view(-1)

        loss = self.loss(logits, labels)
        # done. it seems unnecessary to use mean
        #  loss = torch.mean(self.loss(logits, labels))

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
    
        log = {'train_loss': 0}
        step_in_epoch = 0
        total_loss = 0
        for step, dataset in enumerate(self.data_loader):
            self.global_step += 1
            step_in_epoch += 1
            sum_features, sum_target, sum_word_lens, sum_ref = self.vocab.summary_to_features(dataset['summaries'])

            sum_features, sum_word_lens, sum_target = Variable(sum_features), Variable(sum_word_lens), Variable(sum_target) 
            #  self.logger.debug(pformat(['sum_features: ', sum_features.data.numpy()]))
            #  self.logger.debug(pformat(['sum_target: ', sum_target.data.numpy()]))
            #  self.logger.debug(['sum_word_lens: ', sum_word_lens])
            #  self.logger.debug(pformat(['sum_ref: ', sum_ref]))
            if self.device is not None:
                sum_features = sum_features.cuda()
                sum_target = sum_target.cuda()
                sum_word_lens = sum_word_lens.cuda()

            self.optimizer.zero_grad()
            tfr = self._update_tfr(epoch)
            if self.use_summaryWriter:
                self.writer.add_scalar('train/tfr', tfr, self.global_step)
            probs, predicts = self.model(sum_features, sum_target, sum_word_lens, tfr)
            loss = self._compute_loss(probs, sum_target[:,1:])
            loss.backward()
            if self.max_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            total_loss += loss.item()

            if self.global_step % self.trainer_config['print_loss_every'] == 0:
                avg_loss = total_loss/self.trainer_config['print_loss_every']
                log['train_loss'] = avg_loss
                self.logger.info('Epoch: %d, global_batch: %d, Batch ID:%d Loss:%f'
                        %(epoch, self.global_step, step_in_epoch, avg_loss))
                if self.use_summaryWriter:
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                total_loss = 0

            if self.global_step % self.trainer_config['print_token_every']== 0:
                hyp = self.vocab.features_to_tokens(predicts.numpy().tolist())
                self.logger.info(['hyp: ', hyp])
                self.logger.info(['ref: ', sum_ref])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # control learning rate gradually smaller

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        #  raise NotImplementedError
        self.model.eval()
        total_val_loss = 0
        val_metrics = []
        METRICS = ["rouge-1", "rouge-2", "rouge-l"]
        STATS = ["f", "p", "r"]   
        final_val_metrics = {m:{s: 0.0 for s in STATS} for m in METRICS}
        with torch.no_grad():
            for step, dataset in enumerate(self.valid_data_loader):
                features, target, sents_len, reference = self.vocab.summary_to_features(dataset['summaries'])
                features, target, sents_len  = Variable(features), Variable(target), Variable(sents_len)
                if self.device is not None:
                    features = features.cuda()
                    target = target.cuda()
                    sents_len = sents_len.cuda()

                probs, predicts = self.model(features, target, sents_len, teacher_forcing_ratio=0)
                loss = self._compute_loss(probs, target[:,1:])
                total_val_loss += loss.item()
                val_metrics.append(self._eval_metrics(predicts, reference))

                if step % self.trainer_config['print_val_token_every']== 0:
                    hyp = self.vocab.features_to_tokens(predicts.numpy().tolist())
                    self.logger.info(['hyp: ', hyp])
                    self.logger.info(['ref: ', reference])


            for i in range(len(val_metrics)):
                for m in METRICS:
                    final_val_metrics[m] = {s: val_metrics[i][m][s] + final_val_metrics[m][s] for s in STATS}
            final_val_metrics = {m: {s: final_val_metrics[m][s] / len(val_metrics) for s in STATS}
                                for m in METRICS}

        self.model.train()
        return {
            'val_loss': total_val_loss / step,
            'val_metrics': final_val_metrics
        }
