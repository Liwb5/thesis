# coding:utf-8
import json
import logging
import random

#  class Document():
#      def __init__(self, content, summary, label):
#          """ label: Sometimes you may using some other methods to extract
#                      which sentences to become ground truth
#          """
#          self.content = content
#          self.summary = summary
#          self.label = label


#  class Dataset():
#      def __init__(self, data_list):
#          self._data = data_list
#
#      def __len__(self):
#          return len(self._data)
#
#      def __call__(self, batch_size, shuffle=True):
#          max_len = len(self._data)
#          if shuffle:
#              random.shuffle(self._data)
#          batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
#          return batchs
#
#      def __getitem__(self, index):
#          return self._data[index]

class Dataset():
    def __init__(self, filepath, data_quota=-1):
        """
        """
        #  logging.info('loading data from json file...')
        self.examples = []
        with open(filepath) as f:
            for idx, line in enumerate(f):
                if idx == data_quota:
                    break
                self.examples.append(json.loads(line))

        #  logging.info('using %d(all) examples to train' % len(self.examples))

            #  self.examples = [json.loads(line) for line in f]

        #  if data_quota == -1:
        #      logging.info('using %d(all) examples to train' % len(self.examples))
        #  elif data_quota >= len(self.examples):
        #      logging.warn('data_quota(%d) is larger than %d(all) examples. Using all examples to train.' % (data_quota, len(self.examples)))
        #  else:
        #      logging.info('using %d examples to train' % data_quota)
        #      self.examples = self.examples[0:data_quota]

    def __getitem__(self, index):
        ex = self.examples[index]
        return ex

    def __len__(self):
        return len(self.examples)
