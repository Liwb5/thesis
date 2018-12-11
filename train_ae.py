# coding:utf8
import sys
import os
import argparse
import logging
import numpy as np
import pickle
import time
import random
sys.path.append('./data_loader')
sys.path.append('./utils')

import torch 
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from Dataset import Document, Dataset
from Vocab import Vocab
from DataLoader import BatchDataLoader, PickleReader 
import models

np.set_printoptions(precision=4, suppress=True)

def test(args):
    try:
        if args.device is not None: 
            torch.cuda.set_device(args.device)

        with open(args.vocab_file, 'rb') as f:
            vocab = pickle.load(f)

        data_loader = PickleReader(args.data_dir)
        data_iter = data_loader.chunked_data_reader('train', data_quota=args.train_example_quota)

        if args.device is not None:
            checkpoint = torch.load(args.load_model)
        else:
            checkpoint = torch.load(args.load_model, map_location=lambda storage, loc: storage)

        # checkpoint['args']['device'] saves the device used as train time
        # if at test time, we are using a CPU, we must override device to None
        if args.device is None:
            checkpoint['args'].device = None

        #  net = getattr(models,checkpoint['args'].model_name)(checkpoint['args'])
        net = getattr(models, checkpoint['args'].model_name)(checkpoint['args'], vocab.embed_matrix)
        net.load_state_dict(checkpoint['model'])

        if args.device is not None: 
            net.cuda()
        net.eval()

        for dataset in data_iter:
            for step, docs in enumerate(BatchDataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
                features, target, _, summaries = vocab.summary_to_features(docs, 
                                                        sent_trunc = args.sent_trunc)

                features, target = Variable(features), Variable(target)
                if args.device is not None:
                    features = features.cuda()
                    target = target.cuda()

                probs, predicts = net(features, target)
                predicted_tokens = vocab.features_to_tokens(predicts.numpy().tolist())
                print ('ref: ')
                print (summaries)
                print ('hyp: ')
                print (predicted_tokens)

    except Exception as e:
        raise


def train(args):
    args.save_dir = ''.join((args.save_dir, args.training_purpose, '/'))
    try:
        os.makedirs(args.save_dir)
        os.makedirs(args.log_dir)
    except OSError:
        if not os.path.isdir(args.save_dir):
            raise
        if not os.path.isdir(args.log_dir):
            raise

    log_file = ''.join((args.log_dir, args.training_purpose, '.log'))
    logging.basicConfig(filename = log_file, filemode = 'w',
            level=logging.DEBUG, format='%(asctime)s : %(filename)s[line:%(lineno)d] : %(levelname)s:  %(message)s')

    try:
        if args.device is not None: 
            torch.cuda.set_device(args.device)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        logging.info('loading vocab')
        with open(args.vocab_file, 'rb') as f:
            vocab = pickle.load(f)

        logging.info('init data loader')
        data_loader = PickleReader(args.data_dir)

        # update args
        args.tgt_vocab_size = vocab.embed_matrix.shape[0]
        args.embed_num = vocab.embed_matrix.shape[0]
        args.embed_dim = vocab.embed_matrix.shape[1]
        args.weights = torch.ones(args.embed_num)
        args.weights[vocab.PAD_ID] = 0
        logging.info('args: %s', args)

        logging.info('building model')
        net = getattr(models, args.model_name)(args, vocab.embed_matrix)

        if args.device is not None: 
            net.cuda()

        params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        net.train()

        logging.info('starting training')
        global_step = 0 
        avg_loss = 0

        for epoch in range(args.epochs):
            train_iter = data_loader.chunked_data_reader('train', data_quota=args.train_example_quota)
            step_in_epoch = 0
            for dataset in train_iter:
                for step, docs in enumerate(BatchDataLoader(dataset, batch_size = args.batch_size, shuffle=True)):
                    global_step += 1
                    step_in_epoch += 1
                    features, target, sents_len, summaries  = vocab.summary_to_features(docs, 
                                                                sent_trunc = args.sent_trunc,) 

                    #  logging.debug(summaries)
                    #  logging.debug(['features size: ', features.size()])
                    #  logging.debug(['target size: ', target.size()])
                    #  logging.debug(['sents_len: ', sents_len])
                    #  tokens = vocab.features_to_tokens(features.numpy().tolist())
                    #  logging.debug(tokens)
                    #  tokens = vocab.features_to_tokens(target.numpy().tolist())
                    #  logging.debug(tokens)
                    time.sleep(5)

                    features, target = Variable(features), Variable(target)
                    if args.device is not None:
                        features = features.cuda()
                        target = target.cuda()

                    probs, _ = net(features,target)
                    loss = net.compute_loss(probs, target[:,:-1])
                    avg_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(net.parameters(), args.max_norm)
                    optimizer.step()

                    if global_step % args.print_every == 0:
                        logging.info('Epoch: %d, global_batch: %d, Batch ID:%d Loss:%f'
                                %(epoch, global_step, step_in_epoch, avg_loss/args.print_every))
                        avg_loss = 0

                    #  if global_step*args.batch_size % args.eval_every == 0:
                    #      val_loss = evaluate(args, net, vocab, criterion)
                    #      logging.info('Epoch: %d, global_batch: %d, Batch ID:%d val_Loss:%f'
                    #              %(epoch, global_step, step_in_epoch, val_loss))

                    if global_step*args.batch_size % args.report_every == 0:
                        logging.info('saving model in %d step' % global_step)
                        net.save()

    except Exception as e:
        logging.exception(e) # record error
        raise


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('-vocab_file', type=str, 
            default='./data/cnn_dailymail_data/finished_dm_data/vocab_file.pickle',) 
            #  default = './data/dm_data_from_summaRuNNer/finished_dm_data/vocab_file.pickle',)
    parser.add_argument('-data_dir', type=str, 
            default='./data/cnn_dailymail_data/finished_dm_data/chunked/',) 
            #  default = './data/dm_data_from_summaRuNNer/finished_dm_data/chunked/',)
    parser.add_argument('-train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=20)      #### mark
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-length_limit', type=int, default=-1,
            help='the max length of the output')
    parser.add_argument('-seed', type=int, default=1667)
    parser.add_argument('-report_every', type=int, default=50000)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-sent_trunc',type=int,default=100)
    parser.add_argument('-doc_trunc',type=int,default=100)
    parser.add_argument('-max_norm',type=float,default=1.0)
    parser.add_argument('-log_dir', type=str, default='logs/')

    # model
    parser.add_argument('-save_dir', type=str, default='checkpoints/')
    parser.add_argument('-model_name',type=str,default='GRU_RuNNer')   #### mark
    parser.add_argument('-embed_num',type=int,default=100000)
    parser.add_argument('-tgt_vocab_size',type=int,default=100000)
    parser.add_argument('-embed_dim',type=int,default=100)
    parser.add_argument('-pos_dim',type=int,default=50)
    parser.add_argument('-pos_num',type=int,default=100)
    parser.add_argument('-seg_num',type=int,default=10)
    parser.add_argument('-hidden_size',type=int,default=200)            #### mark
    parser.add_argument('-weights',type=int,default=10)

    #device
    parser.add_argument('-device',type=int)    #### mark

    # test
    parser.add_argument('-load_model',type=str,default='checkpoints/')  #### mark
    parser.add_argument('-topk',type=int,default=3)
    parser.add_argument('-output_dir',type=str,default='outputs/')

    # option
    parser.add_argument('-training_purpose', type=str, default='first_train')  #### mark
    parser.add_argument('-test',action='store_true')
    parser.add_argument('-debug', action='store_true', default=False)

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)
