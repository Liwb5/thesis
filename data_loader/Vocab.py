# coding:utf-8
import numpy as np
import logging
import random
import pickle
import json
import torch
from torch.autograd import Variable

class Vocab():
    def __init__(self, vocab_file, doc_trunc=50, sent_trunc=120, split_token='\n', embed=None):

        with open(vocab_file) as f:
            self.word2id = json.load(f)

        self.id2word = {v:k for k,v in self.word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.SOS_TOKEN = 'SOS_TOKEN'
        self.EOS_TOKEN = 'EOS_TOKEN'
        self.embedding = embed
        self.vocab_size = len(self.word2id)
        self.doc_trunc = doc_trunc
        self.sent_trunc = sent_trunc
        self.split_token = split_token

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

    def data_to_features(self, data):
        docs = data['doc']
        summaries = data['summaries']
        labels = data['labels']
        docs_features, sents_list, doc_lens = self.docs_to_features(docs)
        summaries_features, summaries_target, summaries_lens, reference = self.summary_to_features(summaries)
        labels, labels_lens = self.process_labels(labels)
        return docs_features, doc_lens, sents_list, summaries_features, summaries_target, summaries_lens, reference, labels, labels_lens

    def process_labels(self, labels):
        if not isinstance(labels,list):
            labels = [labels]

        targets = []
        max_len = 0
        labels_lens = []
        for label in labels:
            label = label.split(self.split_token)
            target = []
            for i, key in enumerate(label):
                if key == '1':
                    target.append(i)
            max_len = max(max_len, len(target))
            labels_lens.append(len(target))
            targets.append(target)

        res = []
        for target in targets:
            feature = target + [self.PAD_ID for _ in range(max_len-len(target))]
            res.append(feature)

        res = torch.LongTensor(res)
        #  labels_lens = torch.LongTensor(labels_lens)

        return res, labels_lens

    def docs_to_features(self, docs):
        if not isinstance(docs,list):
            docs = [docs]

        # truncate document 
        sents_list = []
        doc_lens = []
        for doc in docs:
            sents = doc.split(self.split_token)
            max_sent_num = min(self.doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents  # here we add different document's sentences in the same list
            doc_lens.append(max_sent_num) # doc_lens to record the sentences number of every document

        # trunc and pad sentence
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > self.sent_trunc:
                words = words[:self.sent_trunc]
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
        #  doc_lens = torch.LongTensor(doc_lens)

        # batch_sents: [['i', 'am', 'a', 'student'],['i', 'am', 'a', 'student']]
        # doc_sent_features: (B, max_sent_len), doc_lens: (B, ) 
        return doc_sent_features, batch_sents, doc_lens

    def sents_to_features(self, sents_list, sents_len, max_sent_len):
        if not isinstance(sents_list, list):
            sents_list = [sents_list]

        input_features = []
        label_features = []
        reference = []
        for idx, sent in enumerate(sents_list):
            feature = [self.w2i(w) for w in sent]
            pad = [self.PAD_ID for _ in range(max_sent_len - sents_len[idx])]
            input_feature = feature + pad
            label_feature = [self.SOS_ID] + feature + [self.EOS_ID] + pad
            input_features.append(input_feature)
            label_features.append(label_feature)
            reference.append(' '.join(sent))
        return input_features, label_features, reference

        
    def summary_to_features(self, summaries):
        if not isinstance(summaries, list):
            summaries = [summaries]

        sents_list = []
        sents_len = []
        max_sent_len = 0
        for summary in summaries:
            sents = summary.split(self.split_token)
            summary = ' '.join(sents) # join to one sentence
            words = summary.split(' ')
            if len(words) > self.sent_trunc:
                words = words[:self.sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            sents_list.append(words)
            sents_len.append(len(words))

        # 让一个batch中的句子按照长度排序
        #  logging.debug(['origin summaries: ', sents_list])
        #  logging.debug(['origin sents_len: ', sents_len])
        #  sents_list = sorted(sents_list, key=lambda x: len(x), reverse = True)
        #  sents_len = sorted(sents_len, reverse = True)
        #  logging.debug(['origin summaries: ', sents_list])
        #  logging.debug(['origin sents_len: ', sents_len])

        input_features, label_features, reference = self.sents_to_features(sents_list, sents_len, max_sent_len)

        input_features = torch.LongTensor(input_features)
        label_features = torch.LongTensor(label_features)
        sents_len = torch.LongTensor(sents_len)

        # sents_len is the words number of summaries. Not just a sentence
        return input_features, label_features, sents_len, reference


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


    def add_embed_from_summaRuNNer(self, embed_file):
        self.embedding = np.load(embed_file)['embedding']
