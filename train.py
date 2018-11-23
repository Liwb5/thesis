# coding:utf8
import sys
import argparse
import logging
import numpy as np
import pickle

import torch 
from torch.autograd import Variable

sys.path.append('./dataLoader')
from Dataset import Document, Dataset
from Vocab import Vocab
from dataLoader import BatchDataLoader, PickleReader 

np.set_printoptions(precision=4, suppress=True)

def train(args):
    use_gpu = args.device is not None 
    if use_gpu:
        torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    print(args)

    print('loading vocab')
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    print('generating config')

    config = Config(
        vocab_size=vocab.embedding.shape[0],
        embedding_dim=vocab.embedding.shape[1],
        position_size=500,
        position_dim=50,
        word_input_size=100,
        sent_input_size=2 * args.hidden,
        word_GRU_hidden_units=args.hidden,
        sent_GRU_hidden_units=args.hidden,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.word2idx,
        id2word=vocab.idx2word,
        dropout=args.dropout,
    )

     model_name = "".join((args.model_file,
                           str(args.ext_model),
                           str(args.rouge_metric), str(args.std_rouge),
                           str(args.rl_baseline_method), "oracle_l", str(args.oracle_length),
                           "bsz", str(args.batch_size), "rl_loss", str(args.rl_loss_method),
                           "train_example_quota", str(args.train_example_quota),
                           "length_limit", str(args.length_limit),
                           "data", os.path.split(args.data_dir)[-1],
                           "hidden", str(args.hidden),
                           "dropout", str(args.dropout),
                           'ext'))
    print('model_name: ', model_name)

    log_name = "".join(("./logs/model",
                         str(args.ext_model),
                         str(args.rouge_metric), str(args.std_rouge),
                         str(args.rl_baseline_method), "oracle_l", str(args.oracle_length),
                         "bsz", str(args.batch_size), "rl_loss", str(args.rl_loss_method),
                         "train_example_quota", str(args.train_example_quota),
                         "length_limit", str(args.length_limit),
                         "hidden", str(args.hidden),
                         "dropout", str(args.dropout),
                         'ext'))

    print('log_name: ', log_name)

    logging.info('init data loader')
    data_loader = PickleReader(args.data_dir)

    logging.info('init extractive model')

    if args.ext_model == "lstm_summarunner":
        extract_net = model.SummaRuNNer(config)
    elif args.ext_model == "gru_summarunner":
        extract_net = model.GruRuNNer(config)
    elif args.ext_model == "bag_of_words":
        extract_net = model.SimpleRuNNer(config)
    elif args.ext_model == "simpleRNN":
        extract_net = model.SimpleRNN(config)
    elif args.ext_model == "RNES":
        extract_net = model.RNES(config)
    elif args.ext_model == "Refresh":
        extract_net = model.Refresh(config)
    elif args.ext_model == "simpleCONV":
        extract_net = model.simpleCONV(config)
    else:
        print("this is no model to load")


    logging.basicConfig(filename = '%s.log' % log_name, 
            level=logging.INFO, format='[%(asctime)s : %(levelname)s]  %(message)s')
    logging.info('config: %s', config)
    logging.info('args: %s', args)


    if use_gpu:
        extract_net.cuda()

    # loss function
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(extract_net.parameters(), lr=args.lr)
    extract_net.train()

    print ('starting training')
    n_step = 100
    for epoch in range(args.epochs_ext):
        train_iter = data_loader.chunked_data_reader('train', data_quota=args.train_example_quota)
        step_in_epoch = 0
        for dataset in train_iter:
            for step, docs in enumerate(BatchDataLoader(dataset, shuffle=True)):
                try:
                    step_in_epoch += 1
                    features, target, _, doc_lens = vocab.docs_to_features(docs)
                    features, target = Variable(features), Variable(target)
                    if use_gpu:
                        features = features.cuda()
                        target = target.cuda()

                    probs = extract_net(features, doc_lens)
                    loss = criterion(probs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    #  clip_grad_norm(extract_net.parameters(), args.max_norm)
                    optimizer.step()

                    if args.debug:
                        print('Batch ID:%d Loss:%f' %(step, loss.data[0]))
                        continue

                    if step % args.print_every == 0:
                        cur_loss = evaluate(extract_net, vocab, val_iter, criterion)
                        if cur_loss < min_loss:
                            min_loss = cur_loss
                            best_path = extract_net.save()
                        logging.info('Epoch: %3d | Min_Val_Loss: %.4f | Cur_Val_Loss: %.4f'\
                                % (epoch, min_loss, cur_loss))



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab_file', type=str, 
            default='./data/cnn_dailymail_data/finished_dm_data/vocab_file.pickle'
            help='the vocabulary of the dataset which contains words and embeddings' )

    parser.add_argument('-data_dir', type=str, 
            default='./data/cnn_dailymail_data/finished_dm_data/chunked/'
            help='the directory of dataset' )

    parser.add_argument('-train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')

    parser.add_argument('-model_file', type=str, 
            default='../checkpoints/model')
    parser.add_argument('-ext_model', type=str, 
            default='BanditSum')
    parser.add_argument('-load_ext', action='store_true', default=False)

    #device
    parser.add_argument('-device',type=int)

    # hyperparameters
    parser.add_argument('-epochs_ext', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-hidden', type=int, default=200)
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-length_limit', type=int, default=-1,
            help='the max length of the output')
    parser.add_argument('-seed', type=int, default=1667)

    #config
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-print_every', type=int, default=10000)


    args = parser.parse_args()


    train(args)
