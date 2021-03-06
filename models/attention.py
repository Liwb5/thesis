# coding:utf-8
import logging
import random
import math
from pprint import pprint, pformat
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim, hidden_dim,
                use_tanh=False, explorate_rate=4, device=None):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)  # project query
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)   # project context
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax= nn.Softmax(dim=1)
        self.use_tanh = use_tanh
        self.device = device
        self.explorate_rate = explorate_rate
        #  self.log_softmax= F.log_softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1 / math.sqrt(hidden_dim), 1 / math.sqrt(hidden_dim))

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        verbosity = True if random.random() < -0.005 else False # for debug
        # (batch, hidden_dim, seq_len)
        # 为了让decoder是hidden size与 context的维度一致
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)

        if verbosity:
            logging.info(['origin attention prob:', att[0]])
        # constrain the scope of att prob, so that after softmax, sample index can do exploration 
        if self.use_tanh:
            att = self.explorate_rate * self.tanh(att)

        if verbosity:
            logging.info(['after tanh attention prob:', att[0]])

        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        alpha = self.softmax(att)
        if verbosity:
            logging.info(['alpha:', alpha[0]])
        #  alpha = F.log_softmax(att, dim=1)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)
        #  logging.debug(['attention init_inf(expected B, max_doc_len[3]): ', pformat(self.inf.data.cpu().numpy())])

