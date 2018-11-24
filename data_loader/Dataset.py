# coding:utf-8

import random

class Document():
    def __init__(self, content, summary, label):
        """ label: Sometimes you may using some other methods to extract
                    which sentences to become ground truth
        """
        self.content = content
        self.summary = summary
        self.label = label


class Dataset():
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self):
        return len(self._data)

    def __call__(self, batch_size, shuffle=True):
        max_len = len(self._data)
        if shuffle:
            random.shuffle(self._data)
        batchs = [self._data[index:index + batch_size] for index in range(0, max_len, batch_size)]
        return batchs

    def __getitem__(self, index):
        return self._data[index]


