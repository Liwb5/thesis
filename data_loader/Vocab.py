# coding:utf-8
import numpy as np
import logging
import random
import pickle
import json
import torch
from torch.autograd import Variable

class Vocab():
    def __init__(self):

        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '<\s>'
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

    def i2w(self, key):
        if key in self.id2word:
            return self.id2word[key]
        else:
            return self.id2word[self.UNK_ID]

    def features_to_tokens(self, features):
        """ @features: (N, L) a list of list sequence. e.g: [[5,6,7,8],[4,9,5,8,9]]
        """
        if not isinstance(features[0], list):
            features = [features]
        #  features = features.tolist()

        tokens = []
        for feature in features:
            token = []
            for i in range(len(feature)):
                #  if feature[i] == self.PAD_ID or i == len(feature)-1:
                #      tokens.append(' '.join(token))
                #      break
                token.append(self.i2w(feature[i]))
            tokens.append(' '.join(token))
        return tokens

    def docs_to_features(self, examples, sent_trunc = 50, doc_trunc=100):
        sents_list, targets, summaries, doc_lens = [], [], [], []
        if not isinstance(examples,list):
            examples = [examples]

        # truncate document 
        for example in examples:
            max_sent_num = min(doc_trunc, len(example.content))
            doc = example.content[:max_sent_num]
            label = example.label[:max_sent_num]
            sents_list += doc # here we add different document's sentences in the same list
            targets += label
            doc_lens.append(len(doc)) # doc_lens to record the sentences number of every document
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
        targets = torch.FloatTensor(targets)

        return doc_sent_features, targets, summaries, doc_lens

    def summary_to_features(self, examples, sent_trunc = 100):
        if not isinstance(examples, list):
            examples = [examples]

        summaries = []
        sents_len = []
        max_sent_len = 0
        for example in examples:
            summary = example.summary
            summary = ' '.join(summary) # join to one sentence
            words = summary.split(' ')
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            summaries.append(words)
            sents_len.append(len(words))

        # 让一个batch中的句子按照长度排序
        #  logging.debug(['origin summaries: ', summaries])
        #  logging.debug(['origin sents_len: ', sents_len])
        summaries = sorted(summaries, key=lambda x: len(x), reverse = True)
        sents_len = sorted(sents_len, reverse = True)
        #  logging.debug(['origin summaries: ', summaries])
        #  logging.debug(['origin sents_len: ', sents_len])

        input_features = []
        label_features = []
        ground_truth = []
        for summary in summaries:
            ground_truth.append(' '.join(summary))
            feature = [self.w2i(w) for w in summary] 
            pad = [self.PAD_ID for _ in range(max_sent_len - len(summary))]
            input_feature = feature + pad  # as input in autoencoder 
            label_feature = [self.SOS_ID] + feature + [self.EOS_ID] + pad  # as label in autoencoder
            input_features.append(input_feature)
            label_features.append(label_feature)

        input_features = torch.LongTensor(input_features)
        label_features = torch.LongTensor(label_features)
        sents_len = torch.LongTensor(sents_len)
        return input_features, label_features, sents_len, ground_truth 


    def add_vocab(self, vocab_file):
        self.word_list = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
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
            eos_embedding = np.random.random((1, embed_size))
            sos_embedding = np.random.random((1, embed_size))
            embed_matrix = np.zeros((len(self.word_list), embed_size))
            embed_matrix[2] = eos_embedding
            embed_matrix[3] = sos_embedding

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
