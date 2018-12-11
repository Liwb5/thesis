# coding:utf-8
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .BasicModule import BasicModule
import torch.nn.utils.rnn as rnn_utils


class rnn_encoder(nn.Module):
    def __init__(self, args, vocab_size, embed=None):
        super(rnn_encoder, self).__init__()

        if embed is not None:
            self.embedding = embed
            #  embed = torch.FloatTensor(embed)
            #  self.embedding.weight.data.copy_(embed)
            #  self.embedding = nn.Embedding.from_pretrained(embed, freeze = True)
            #  self.embedding.weight.requires_grad=True
        else:
            self.embedding = nn.Embedding(vocab_size, args.embed_dim)

        self.args = args
        self.use_cuda = True if args.device is not None else False

        D = args.embed_dim
        H = args.hidden_size

        self.RNN = nn.GRU(
                        input_size = D, 
                        hidden_size = H, 
                        batch_first = False,
                        bidirectional = True)

    def avg_pool1d(self, x, seq_lens):
        """ @x: (n, l, h).  average pooling in second dimension (l)
            @seq_lens: (n, 1)  
            @output: (n, h)
        """
        output = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :] # (l_n, h)
            t = torch.t(t).unsqueeze(0) # (1, h, l_n)
            output.append(f.avg_pool1d(t, t.size(2)))

        output = torch.cat(output).squeeze(2)  
        return output    # (n, h)

    def forward(self, x):
        """ @x: (B, seq_len). 
        """
        sent_lens = torch.sum(torch.sign(x), dim=1) # (N, 1). the real length of every sentence
        x = self.embedding(x).transpose(0, 1)    # (L, B, E). E: embedding dimension
        packed = rnn_utils.pack_padded_sequence(x,lengths = list(sent_lens))

        enc_out, h_n = self.RNN(packed)   # output: (L, B, 2*H) h_n: (num_layer*num_direction, B, H)
        enc_out, _ = rnn_utils.pad_packed_sequence(enc_out)

        return enc_out.transpose(0, 1), h_n
