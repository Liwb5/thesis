# coding:utf8
import sys
import os
import argparse
import logging
import numpy as np
import pickle
import time
import random
from tqdm import tqdm
sys.path.append('./data_loader')
sys.path.append('./utils')

import torch 
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from Dataset import Document, Dataset
from Vocab import Vocab
from DataLoader import BatchDataLoader, PickleReader 
from helper import Config
import models

np.set_printoptions(precision=4, suppress=True)

def evaluate(args, net, vocab, criterion):
    data_loader = PickleReader(args.data_dir)
    val_iter = data_loader.chunked_data_reader('val', data_quota=args.train_example_quota)
    net.eval()
    total_loss = 0
    batch_num = 0
    for dataset in val_iter:
        for docs in BatchDataLoader(dataset, batch_size = args.batch_size, shuffle=False):
            features, target, _, doc_lens = vocab.docs_to_features(docs, 
                                                                sent_trunc = args.sent_trunc, 
                                                                doc_trunc = args.doc_trunc)
            features, target = Variable(features), Variable(target)
            if args.device is not None:
                features = features.cuda()
                target = target.cuda()

            probs = extract_net(features, doc_lens)
            loss = criterion(probs, target)
            total_loss += loss.item()
            #  del loss   # if not apply gradient. then the dynamic graph will stay in free. should delete manually
            batch_num += 1

    loss = total_loss / batch_num
    net.train()
    return loss

def test(args):
    try:
        if args.device is not None: 
            torch.cuda.set_device(args.device)

        logging.info('loading vocab')
        with open(args.vocab_file, 'rb') as f:
            vocab = pickle.load(f)

        logging.info('init data loader')
        data_loader = PickleReader(args.data_dir)
        test_iter = data_loader.chunked_data_reader('test')

        if args.device is not None:
            checkpoint = torch.load(args.load_dir)
        else:
            checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

        # checkpoint['args']['device'] saves the device used as train time
        # if at test time, we are using a CPU, we must override device to None
        if args.device is None:
            checkpoint['args'].device = None

        net = getattr(models,checkpoint['args'].model_name)(checkpoint['args'])
        net.load_state_dict(checkpoint['model'])

        if args.device is not None: 
            net.cuda()
        net.eval()

        args.output_dir = args.output_dir + checkpoint['args'].training_purpose + '/'
        try:
            os.makedirs(args.output_dir + 'ref/')
            os.makedirs(args.output_dir + 'hyp/')
        except OSError:
            if not os.path.isdir(args.output_dir + 'ref/'):
                raise
            if not os.path.isdir(args.output_dir + 'hyp/'):
                raise

        file_id = 0
        for dataset in tqdm(test_iter):
            for step, docs in enumerate(BatchDataLoader(dataset, batch_size = checkpoint['args'].batch_size, shuffle=False)):
                features, _, summaries, doc_lens = vocab.docs_to_features(docs,
                                                                sent_trunc = args.sent_trunc, 
                                                                doc_trunc = args.doc_trunc)
                if args.device is not None:
                    probs = net(Variable(features).cuda(), doc_lens)
                else:
                    probs = net(Variable(features), doc_lens)

                start = 0
                for doc_id, doc_len in enumerate(doc_lens):
                    end = start + doc_len
                    prob = probs[start:end]
                    topk = min(args.topk, doc_len)
                    topk_indices = prob.topk(topk)[1].cpu().data.numpy()
                    topk_indices.sort()
                    article = docs[doc_id].content
                    hyp = [article[index] for index in topk_indices]
                    ref = summaries[doc_id]
                    with open(os.path.join(args.output_dir+'ref/',str(file_id)+'.txt'), 'w') as f:
                        f.write('\n'.join(ref))
                    with open(os.path.join(args.output_dir+'hyp/',str(file_id)+'.txt'), 'w') as f:
                        f.write('\n'.join(hyp))

                    start = end
                    file_id += 1
                
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
        args.embed_num = vocab.embed_matrix.shape[0]
        args.embed_dim = vocab.embed_matrix.shape[1]
        logging.info('args: %s', args)

        logging.info('init extractive model')
        extract_net = getattr(models, args.model_name)(args, vocab.embed_matrix)
        #  if args.model_name == "lstm_summarunner":
        #      extract_net = SummaRuNNer(args)
        #  elif args.model_name == "GRU_RuNNer":
        #      extract_net = GRU_RuNNer(args, vocab.embed_matrix)
        #  elif args.model_name == "bag_of_words":
        #      extract_net = SimpleRuNNer(args)
        #  elif args.model_name == "simpleRNN":
        #      extract_net = SimpleRNN(args)
        #  elif args.model_name == "RNES":
        #      extract_net = RNES(args)
        #  elif args.model_name == "Refresh":
        #      extract_net = Refresh(args)
        #  elif args.model_name == "simpleCONV":
        #      extract_net = simpleCONV(args)
        #  else:
        #      logging.error("There is no model to load")
        #      sys.exit()

        if args.device is not None: 
            extract_net.cuda()

        # loss function
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(extract_net.parameters(), lr=args.lr)
        extract_net.train()

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
                    features, target, _, doc_lens = vocab.docs_to_features(docs,
                                                                sent_trunc = args.sent_trunc, 
                                                                doc_trunc = args.doc_trunc)
                    #  logging.debug(features)
                    #  logging.debug(target)
                    #  time.sleep(5)
                    features, target = Variable(features), Variable(target)
                    if args.device is not None:
                        features = features.cuda()
                        target = target.cuda()

                    probs = extract_net(features, doc_lens)
                    #  logging.debug(probs)
                    loss = criterion(probs, target)
                    avg_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(extract_net.parameters(), args.max_norm)
                    optimizer.step()

                    if global_step % args.print_every == 0:
                        logging.info('Epoch: %d, global_batch: %d, Batch ID:%d Loss:%f' 
                                %(epoch, global_step, step_in_epoch, avg_loss/args.print_every))
                        avg_loss = 0

                    if global_step*args.batch_size % args.eval_every == 0:
                        val_loss = evaluate(args, extract_net, vocab, criterion)
                        logging.info('Epoch: %d, global_batch: %d, Batch ID:%d val_Loss:%f'
                                %(epoch, global_step, step_in_epoch, val_loss))

                    if global_step*args.batch_size % args.report_every == 0:
                        logging.info('saving model in %d step' % global_step)
                        extract_net.save()

    except Exception as e:
        logging.exception(e) # record error
        raise



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('-vocab_file', type=str, 
            #  default='./data/cnn_dailymail_data/finished_dm_data/vocab_file.pickle',
            default = './data/dm_data_from_summaRuNNer/finished_dm_data/vocab_file.pickle',)
    parser.add_argument('-data_dir', type=str, 
            #  default='./data/cnn_dailymail_data/finished_dm_data/chunked/',
            default = './data/dm_data_from_summaRuNNer/finished_dm_data/chunked/',)
    parser.add_argument('-train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=20)      #### mark
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-length_limit', type=int, default=-1,
            help='the max length of the output')
    parser.add_argument('-seed', type=int, default=1667)
    parser.add_argument('-report_every', type=int, default=100000)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument('-eval_every', type=int, default=10000)
    parser.add_argument('-sent_trunc',type=int,default=50)
    parser.add_argument('-doc_trunc',type=int,default=100)
    parser.add_argument('-max_norm',type=float,default=1.0)
    parser.add_argument('-log_dir', type=str, default='logs/')

    # model
    parser.add_argument('-save_dir', type=str, default='checkpoints/')
    parser.add_argument('-model_name',type=str,default='GRU_RuNNer')   #### mark
    parser.add_argument('-embed_num',type=int,default=100000)
    parser.add_argument('-embed_dim',type=int,default=100)
    parser.add_argument('-pos_dim',type=int,default=50)
    parser.add_argument('-pos_num',type=int,default=100)
    parser.add_argument('-seg_num',type=int,default=10)
    parser.add_argument('-hidden_size',type=int,default=200)            #### mark


    #device
    parser.add_argument('-device',type=int)    #### mark

    # test
    parser.add_argument('-load_dir',type=str,default='checkpoints/')  #### mark
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
