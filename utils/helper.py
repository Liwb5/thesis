# coding:utf-8
import pickle
import logging
import random
from collections import namedtuple

import numpy as np
import torch 
from torch.autograd import Variable


Config = namedtuple('parameters',
                    ['vocab_size', 'embedding_dim',
                     'position_size', 'position_dim',
                     'word_input_size', 'sent_input_size',
                     'word_GRU_hidden_units', 'sent_GRU_hidden_units',
                     'pretrained_embedding', 'word2id', 'id2word',
                     'dropout'])


