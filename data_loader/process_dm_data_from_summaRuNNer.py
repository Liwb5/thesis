# coding:utf-8
import argparse
import hashlib
import pickle, tarfile
import random
import re
import os
import numpy as np
import json
from Dataset import Document, Dataset
from Vocab import Vocab

def build_dataset(args):

    def write_to_pickle(source_file, target_dir, chunk_size=10000):
        with open(source_file) as f:
            examples = [json.loads(line) for line in f]

        chunked_data = []
        for i, example in enumerate(examples):
            if i % chunk_size == 0 and i > 0:
                pickle.dump(Dataset(chunked_data), open(target_dir % (i / chunk_size), 'wb'))
                print ('%d samples have been processed.' % (i))
                chunked_data = []

            doc = example['doc'].split('\n')
            label = example['labels'].split('\n')
            label = [int(item) for item in label]
            summary = example['summaries'].split('\n')
            chunked_data.append(Document(doc, summary, label))

        if chunked_data != []:
            pickle.dump(Dataset(chunked_data), open(target_dir % (i / chunk_size + 1), 'wb'))
            print ('%d samples have been processed.' % (i))

    print (args)
    print('start building Dataset')
    train_file = args.source_dir + 'train.json'
    test_file = args.source_dir + 'test.json'
    val_file = args.source_dir + 'val.json'

    target_dir = args.target_dir + 'chunked/'
    try:
        os.makedirs(target_dir)
    except OSError:
        if not os.path.isdir(target_dir):
            print ('can not create target_dir! ')
            exit()

    write_to_pickle(train_file,
            ''.join((target_dir, 'train_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(test_file, 
            ''.join((target_dir, 'test_%03d.pickle')),
            chunk_size = 10000)

    write_to_pickle(val_file, 
            ''.join((target_dir, 'val_%03d.pickle')),
            chunk_size = 10000)


def build_vocab(args):
    print(args)
    print('starting building vocab')

    vocab = Vocab()

    print('starting loading word2id')
    vocab.add_vocab_from_summaRuNNer(args.vocab_file)
    print('starting loading embeddings')
    vocab.add_embed_from_summaRuNNer(args.embed_file)

    print('finished building vocab')
    pickle.dump(vocab, open(args.finished_vocab, 'wb'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-source_dir', type=str, 
            default='../data/dm_data_from_summaRuNNer/origin_data/')
    parser.add_argument('-target_dir', type=str, default='../data/dm_data_from_summaRuNNer/finished_dm_data/')

    # if you want to build vocab, below are some parameters that you should considern
    parser.add_argument('-build_vocab', action='store_true', default=False)
    parser.add_argument('-vocab_file', type=str, default='../data/dm_data_from_summaRuNNer/origin_data/word2id.json')
    parser.add_argument('-embed_file', type=str, default='../data/dm_data_from_summaRuNNer/origin_data/embedding.npz')
    #  parser.add_argument('-embed_size', type=int, default=100)
    parser.add_argument('-finished_vocab', type=str, default='../data/dm_data_from_summaRuNNer/finished_dm_data/vocab_file.pickle')

    args = parser.parse_args()

    if args.build_vocab:
        build_vocab(args)
    else:
        build_dataset(args)
