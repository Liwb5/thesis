# coding:utf-8
import numpy as np
import random
import pickle
import json

class Vocab():
    def __init__(self):

        self.PAD_ID = 0
        self.UNK_ID = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.word_list = []
        self.word2id = {}
        self.id2word = {}
        self.count = 0
        self.embed_matrix = None

    def __getitem__(self, key):
        if self.word2id.has_key(key):
            return self.word2id[key]
        else:
            return self.word2id[self.UNK_TOKEN]


    def add_vocab(self, vocab_file):
        self.word_list = [self.PAD_TOKEN, self.UNK_TOKEN, '<s>', '<\s>']
        with open(vocab_file, 'r') as f:
            for line in f:
                self.word_list.append(line.split()[0]) # only want the word, not the count
            print ('read %d words from vocab file' % len(self.word_list))

        for w in self.word_list:
            self.word2id[w] = self.count
            self.id2word[self.count] = w
            self.count += 1

    def add_embedding(self, embed_file, embed_size):
        print('Loading embeddings ')
        with open(embed_file, 'r') as f:
            word_set = set(self.word_list)
            embed_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            count = 0
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in word_set:
                    count += 1
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    embed_matrix[self.word2id[word]] = embedding

                if count % 10000 == 0:
                    print('processed %d data' % count)

            self.embed_matrix = embed_matrix
            print('%d words out of %d has embeddings in the embed_file' % (count, len(self.word_list)))


    def add_vocab_from_summaRuNNer(self, vocab_file):
        with open(vocab_file) as f:
            self.word2id = json.load(f)

        self.id2word = {v:k for k , v in self.word2id.items()}
        assert len(self.id2word) == len(self.word2id)

    def add_embed_from_summaRuNNer(self, embed_file):
        self.embed_matrix = np.load(embed_file)['embedding']
