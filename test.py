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
from tqdm import tqdm
from pprint import pprint, pformat

import torch
from torch.utils.data import DataLoader
import models
import models.loss as module_loss
import models.metrics as module_metrics
#  from trainer import Trainer, AE_trainer
import trainer as module_trainer
from utils import Logger
from utils import StreamToLogger
from utils.config import *
from data_loader import Vocab
from data_loader import Dataset

class Test():
    def __init__(self, model, loss, config, test_data_loader, metrics, vocab):
        self.model = model
        self.loss = loss
        self.config = config
        self.test_data_loader = test_data_loader
        self.metrics = metrics
        self.vocab = vocab

    def _eval_metrics(self, docs, pointers, reference):
        #  hypothesis = self.vocab.features_to_tokens(predicts.numpy().tolist())
        hypothesis = self.vocab.extract_summary_from_index(docs, pointers)
        results = self.metrics(hypothesis, reference)
        return results

    def test(self):
        """
        testing data to get result
        """
        self.model.eval()
        total_test_loss = 0
        test_metrics = []
        METRICS = ["rouge-1", "rouge-2", "rouge-l"]
        STATS = ["f", "p", "r"]   
        final_test_metrics = {m:{s: 0.0 for s in STATS} for m in METRICS}
        with torch.no_grad():
            for step, dataset in enumerate(self.test_data_loader):
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

                att_probs, selected_probes, pointers, _ = self.model(docs_features, doc_lens, sum_features, sum_word_lens, labels, label_lens, tfr=0)

                test_metrics.append(self._eval_metrics(dataset['doc'], pointers, sum_ref))

            for i in range(len(test_metrics)):
                for m in METRICS:
                    final_test_metrics[m] = {s: test_metrics[i][m][s] + final_test_metrics[m][s] for s in STATS}

            final_test_metrics = {m: {s: final_test_metrics[m][s] / len(test_metrics) for s in STATS}
                                for m in METRICS}

        return {'task_name': self.config['task_name'], 
            'test_metrics': final_test_metrics}

def test(config, resume):
    test_data = Dataset(config['data_loader']['test_data'], data_quota = -1)
    test_data_loader = DataLoader(dataset = test_data,
                            batch_size = config['data_loader']['batch_size'])
    logging.info('using %d examples to test. ' % len(test_data))

    vocab = Vocab(**config['vocabulary'], embed=None)
    # build model architecture
    model = getattr(models, config['model']['type'])(config['model']['args'], device=config['device'])
    logging.info(['model infomation: ', model])

    # get function handles of loss and metrics
    weights = torch.ones(config['model']['args']['vocab_size'])
    weights[vocab.PAD_ID] = 0
    loss = getattr(module_loss, config['loss'])(weights)

    metrics = getattr(module_metrics, config['metrics']) 

    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    if config['device'] is not None:
        model = model.cuda()

    t = Test(model, loss, config, test_data_loader, metrics, vocab)
    result = t.test()
    logging.info(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='thesis_test')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    test(config, args.resume)
