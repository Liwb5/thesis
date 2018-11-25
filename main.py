# coding:utf8
import sys
import argparse
import logging
import numpy as np
import pickle
import time
import random
sys.path.append('./data_loader')
sys.path.append('./utils')
sys.path.append('./models')

import torch 
import torch.nn as nn
from torch.autograd import Variable
from Dataset import Document, Dataset
from Vocab import Vocab
from DataLoader import BatchDataLoader, PickleReader 
from helper import Config
from GRU_RuNNer import GRU_RuNNer


np.set_printoptions(precision=4, suppress=True)

def train(args):
    args.save_dir = ''.join((args.save_dir, args.training_purpose, '/'))
    args.log_dir = ''.join((args.log_dir, args.training_purpose))

    logging.basicConfig(filename = '%s.log' % args.log_dir, 
            filemode = 'w',
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

        if args.model_name == "lstm_summarunner":
            extract_net = SummaRuNNer(args)
        elif args.model_name == "GRU_RuNNer":
            extract_net = GRU_RuNNer(args, vocab.embed_matrix)
        elif args.model_name == "bag_of_words":
            extract_net = SimpleRuNNer(args)
        elif args.model_name == "simpleRNN":
            extract_net = SimpleRNN(args)
        elif args.model_name == "RNES":
            extract_net = RNES(args)
        elif args.model_name == "Refresh":
            extract_net = Refresh(args)
        elif args.model_name == "simpleCONV":
            extract_net = simpleCONV(args)
        else:
            logging.error("There is no model to load")
            sys.exit()

        if args.device is not None: 
            extract_net.cuda()

        # loss function
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(extract_net.parameters(), lr=args.lr)
        extract_net.train()

        logging.info('starting training')
        n_step = 100
        for epoch in range(args.epochs):
            train_iter = data_loader.chunked_data_reader('train', data_quota=args.train_example_quota)
            step_in_epoch = 0
            for dataset in train_iter:
                for step, docs in enumerate(BatchDataLoader(dataset, shuffle=True)):
                    step_in_epoch += 1
                    features, target, _, doc_lens = vocab.docs_to_features(docs)
                    #  logging.debug(features)
                    #  logging.debug(target)
                    time.sleep(5)
                    features, target = Variable(features), Variable(target)
                    if args.device is not None:
                        features = features.cuda()
                        target = target.cuda()

                    probs = extract_net(features, doc_lens)
                    logging.debug(probs.data)
                    #  loss = criterion(probs, target)
                    #  optimizer.zero_grad()
                    #  loss.backward()
                    #  #  clip_grad_norm(extract_net.parameters(), args.max_norm)
                    #  optimizer.step()
                    #
                    #  if args.debug:
                    #      print('Batch ID:%d Loss:%f' %(step, loss.data[0]))
                    #      continue
                    #
                    #  if step % args.print_every == 0:
                    #      cur_loss = evaluate(extract_net, vocab, val_iter, criterion)
                    #      if cur_loss < min_loss:
                    #          min_loss = cur_loss
                    #          best_path = extract_net.save()
                    #      logging.info('Epoch: %3d | Min_Val_Loss: %.4f | Cur_Val_Loss: %.4f'\
                    #              % (epoch, min_loss, cur_loss))
    except Exception as e:
        logging.exception(e) # record error
        raise



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('-vocab_file', type=str, 
            #  default='./data/cnn_dailymail_data/finished_dm_data/vocab_file.pickle',
            default = './data/dm_data_from_summaRuNNer/finished_dm_data/vocab_file.pickle',
            help='the vocabulary of the dataset which contains words and embeddings')
    parser.add_argument('-data_dir', type=str, 
            #  default='./data/cnn_dailymail_data/finished_dm_data/chunked/',
            default = './data/dm_data_from_summaRuNNer/finished_dm_data/chunked/',
            help='the directory of dataset' )
    parser.add_argument('-train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-length_limit', type=int, default=-1,
            help='the max length of the output')
    parser.add_argument('-seed', type=int, default=1667)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-print_every', type=int, default=10000)
    parser.add_argument('-seq_trunc',type=int,default=50)
    parser.add_argument('-max_norm',type=float,default=1.0)
    parser.add_argument('-training_purpose', type=str, default='first_train')
    parser.add_argument('-log_dir', type=str, default='logs/')

    # model
    parser.add_argument('-save_dir', type=str, default='checkpoints/')
    parser.add_argument('-model_name',type=str,default='GRU_RuNNer')
    parser.add_argument('-embed_num',type=int,default=100000)
    parser.add_argument('-embed_dim',type=int,default=100)
    parser.add_argument('-pos_dim',type=int,default=50)
    parser.add_argument('-pos_num',type=int,default=100)
    parser.add_argument('-seg_num',type=int,default=10)
    parser.add_argument('-hidden_size',type=int,default=200)
    parser.add_argument('-kernel_num',type=int,default=100) # for CNN
    parser.add_argument('-kernel_sizes',type=str,default='3,4,5')  # for CNN


    #device
    parser.add_argument('-device',type=int)

    args = parser.parse_args()

    train(args)
