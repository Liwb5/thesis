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
        self.max_norm = self.trainer_config['max_norm']
        #  self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, docs, pointers, reference):
        #  hypothesis = self.vocab.features_to_tokens(predicts.numpy().tolist())
        hypothesis = self.vocab.extract_summary_from_index(docs, pointers)
        results = self.metrics(hypothesis, reference)
        return results
        
    def _compute_reward(self, dataset, pointers, ref):
        R = []
        batch_size = pointers.size(0)
        for i in range(pointers.size(1)):
            hyp = self.vocab.extract_summary_from_index(dataset['doc'], pointers[:,:i+1].view(batch_size, -1))
            self.logger.debug(pformat(['hyp: ', hyp]))
            result = self.metrics(hyp, ref, avg=False)
            # maybe can divide to 3
            r = [item['rouge-1']['f'] + item['rouge-2']['f'] + item['rouge-l']['f'] \
                for item in result]
            R.append(r)
        
        R = torch.FloatTensor(R).transpose(0,1)
        final_R = R[:,-1]
        R = R - torch.cat((torch.zeros((batch_size,1)), R[:,:-1].view(batch_size,-1)), 1)
        R = Variable(R, requires_grad=False)
        if self.device is not None:
            R = R.cuda()
        return R, final_R

    def _compute_only_final_reward(self, dataset, pointers, refs):
        hyps = self.vocab.extract_summary_from_index(dataset['doc'], pointers)
        self.logger.debug(pformat(['hyps: ', hyps]))
        result = self.metrics(hyps, refs, avg=False)
        r = [item['rouge-1']['f'] + item['rouge-2']['f'] + item['rouge-l']['f'] \
            for item in result]
        r = torch.FloatTensor(r).view(-1,1)
        R = r.repeat(1, pointers.size(1))
        #  self.logger.debug(pformat(['R: ', R]))
        R = Variable(R, requires_grad=False)
        if self.device is not None:
            R = R.cuda()
            r = r.cuda()
        return R, r



    #  def predict(self, predicts, target):
    #      """
    #      @ target: Varialbe, (B, L)
    #      """
    #      predicted_tokens = self.vocab.features_to_tokens(predicts.numpy().tolist())
    #      target_tokens = self.vocab.features_to_tokens(target.data.cpu().numpy().tolist())
    #      self.logger.info(['hyp: ', predicted_tokens])
    #      self.logger.info(['ref: ', target_tokens])

    def _update_tfr(self):
        tfr = max(math.exp(-(self.global_step)/200000-0.1), 0.5)
        #  tfr = self.trainer_config['teacher_forcing_ratio'] - self.global_step/len(self.data_loader)
        return tfr 

    def _update_e_greedy(self):
        return self.trainer_config['epsilon']

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


    def _compute_loss2(self, probs, R):
        num_samples = R.size(0) * R.size(1)
        logprobs = torch.log(probs)
        loss = torch.mul(logprobs, R).view(-1)
        loss = -loss.sum() / num_samples * 1000
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
        total_reward = 0
        for step, dataset in enumerate(self.data_loader):
            self.global_step += 1
            step_in_epoch += 1
            #  features, target, sents_len, reference = self.vocab.summary_to_features(dataset['summaries'])
            docs_features, doc_lens, docs_tokens, \
                    sum_features, sum_target, sum_word_lens, sum_ref, \
                    labels, label_lens = self.vocab.data_to_features(dataset)

            docs_features = Variable(docs_features) 
            sum_features, sum_word_lens, sum_target = Variable(sum_features), Variable(sum_word_lens), Variable(sum_target) 
            #  labels = Variable(labels)
            #  self.logger.debug(pformat(['docs_features: ', docs_features.data.numpy()]))
            #  self.logger.debug(pformat(['docs_tokens: ', docs_tokens]))
            self.logger.debug(['doc_lens: ', doc_lens])
            #  self.logger.debug(pformat(['sum_features: ', sum_features.data.numpy()]))
            #  self.logger.debug(pformat(['sum_target: ', sum_target.data.numpy()]))
            #  self.logger.debug(['sum_word_lens: ', sum_word_lens])
            #  self.logger.debug(pformat(['labels: ', labels.data.numpy()]))
            #  self.logger.debug(['label_lens: ', label_lens])
            if self.device is not None:
                docs_features = docs_features.cuda()
                #  doc_lens = doc_lens.cuda()
                sum_features = sum_features.cuda()
                sum_target = sum_target.cuda()
                sum_word_lens = sum_word_lens.cuda()
                #  labels = labels.cuda()
                #  label_lens = label_lens.cuda()

            self.optimizer.zero_grad()
            tfr = self._update_tfr()
            epsilon = self._update_e_greedy()
            if self.use_summaryWriter:
                self.writer.add_scalar('train/tfr', tfr, self.global_step)
                self.writer.add_scalar('train/epsilon', epsilon, self.global_step)
            att_probs, selected_probs, pointers, multi_indices = self.model(docs_features, doc_lens, sum_features, sum_word_lens, labels, label_lens, tfr, epsilon = epsilon)

            self.logger.debug(pformat(['multi_indices: ', multi_indices]))
            multi_sample_reward = []
            for indices in multi_indices:
                _, final_R = self._compute_only_final_reward(dataset, indices, sum_ref)
                multi_sample_reward.append(final_R.unsqueeze(0))
            multi_sample_reward = torch.cat(multi_sample_reward)  # (sample_num, B, 1)
            avg_sample_reward = multi_sample_reward.mean(0)    #(B,1)
            self.logger.debug(pformat(['multi_sample_reward: ', multi_sample_reward]))
            self.logger.debug(pformat(['avg_sample_reward: ', avg_sample_reward]))

            self.logger.debug(pformat(['selected_probs: ', selected_probs]))
            self.logger.debug(pformat(['pointers: ', pointers]))
            self.logger.debug(pformat(['sum_ref: ', sum_ref]))
            R, final_R = self._compute_only_final_reward(dataset, pointers, sum_ref)
            self.logger.debug(pformat(['R: ', final_R]))
            advantage_R = R - avg_sample_reward
            self.logger.debug(pformat(['advantage_R: ', advantage_R]))
            loss = self._compute_loss2(selected_probs, advantage_R)
            #  loss = self._compute_loss()
            self.logger.debug(pformat(['loss: ', loss.item()]))

            # DONE how to compute reward. see RL_combin how to do it 
            #  selected_docs_features = docs_features[pred_index.byte().data].view(docs_features.size(0), pred_index.size(1))
            #  R = self.eval_model.compute_reward(docs_features, summaries_features)
            #  loss = self._compute_loss(probs, target[:,1:])
            loss.backward()
            if self.max_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            total_loss += loss.item()
            total_reward += final_R.sum().item()

            if self.global_step % self.trainer_config['print_loss_every'] == 0:
                avg_loss = total_loss/self.trainer_config['print_loss_every']
                avg_reward = total_reward/self.trainer_config['print_loss_every']/self.batch_size
                self.logger.info('Epoch: %d, global_batch: %d, Batch ID:%d Loss:%f Reward: %f'
                        %(epoch, self.global_step, step_in_epoch, avg_loss, avg_reward))
                if self.use_summaryWriter:
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/rewards', avg_reward, self.global_step)
                total_loss = 0
                total_reward = 0

            #  if self.global_step % self.trainer_config['print_token_every']== 0:
            #      hyp = self.vocab.features_to_tokens(predicts.numpy().tolist())
            #      self.logger.info(['hyp: ', hyp])
            #      self.logger.info(['ref: ', reference])

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # control learning rate gradually smaller

        log = {}
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
                docs_features, doc_lens, docs_tokens, \
                    sum_features, sum_target, sum_word_lens, sum_ref, \
                    labels, label_lens = self.vocab.data_to_features(dataset)

                docs_features = Variable(docs_features) 
                sum_features, sum_word_lens, sum_target = Variable(sum_features), Variable(sum_word_lens), Variable(sum_target) 
                labels = Variable(labels)

                if self.device is not None:
                    docs_features = docs_features.cuda()
                    #  doc_lens = doc_lens.cuda()
                    sum_features = sum_features.cuda()
                    sum_target = sum_target.cuda()
                    sum_word_lens = sum_word_lens.cuda()
                    labels = labels.cuda()
                    #  label_lens = label_lens.cuda()

                att_probs, selected_logprobes, pointers, _ = self.model(docs_features, doc_lens, sum_features, sum_word_lens, labels, label_lens, tfr=0)

                val_metrics.append(self._eval_metrics(dataset['doc'], pointers, sum_ref))

                #  if step % self.trainer_config['print_val_token_every']== 0:
                #      hyp = self.vocab.features_to_tokens(predicts.numpy().tolist())
                #      self.logger.info(['hyp: ', hyp])
                #      self.logger.info(['ref: ', reference])


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
