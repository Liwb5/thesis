# coding:utf-8
import numpy as np
import random
import pickle
import json
import torch
from torch.autograd import Variable

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

    def w2i(self, key):
        if key in self.word2id:
            return self.word2id[key]
        else:
            return self.word2id[self.UNK_TOKEN]

    def docs_to_features(self, examples, sent_trunc = 50, doc_trunc=100):
        sents_list, targets, summaries, doc_lens = [], [], [], []
        if not isinstance(examples,list):
            examples = [examples]

        # truncate document 
        for example in examples:
            max_sent_num = min(doc_trunc, len(example.content))
            doc = example.content[:max_sent_num]
            label = example.label[:max_sent_num]
            sents_list += doc
            targets += label
            doc_lens.append(len(doc))
            summaries.append(example.summary)

        # trunc and pad sentence
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + \
                    [self.PAD_ID for _ in range(max_sent_len - len(sent))]
            features.append(feature)
            
        #  doc_sent_features = []
        #  doc_sent_features.append(features[:doc_lens[0]])
        #  for i in range(len(doc_lens)-1):
        #      doc_sent_features.append(features[doc_lens[i]:doc_lens[i+1]])

        doc_sent_features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)

        return doc_sent_features, targets, summaries, doc_lens


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
