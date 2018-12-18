# coding:utf-8
import torch.nn as nn
import torch.nn.functional as F

def ce_loss(weights):
    return nn.CrossEntropyLoss(weight = weights)
