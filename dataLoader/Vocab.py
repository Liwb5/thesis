# coding:utf-8
import numpy as np
import random
import pickle

class Vocab():
    def __init__(self):

        #  PAD_ID = 0
        #  UNK_ID = 1
        #  PAD_TOKEN = '<pad>'
        #  UNK_TOKEN = '<unk>'
        self.word_list = ['<pad>', '<unk>', '<s>', '<\s>']
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0
        self.embedding = None

    def __getitem__(self, key):
        if self.word2idx.has_key(key):
            return self.word2idx[key]
        else:
            return self.word2idx['<unk>']


    def add_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            for line in f:
                self.word_list.append(line.split()[0]) # only want the word, not the count
            print ('read %d words from vocab file' % len(self.word_list))

        for w in self.word_list:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
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
                    embed_matrix[self.word2idx[word]] = embedding

                if count % 10000 == 0:
                    print('processed %d data' % count)

            self.embeddings = embed_matrix
            print('%d words out of %d has embeddings in the embed_file' % (count, len(self.word_list)))


