# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys
import logging
sys.path.append('../')
from seq2seq.models import DecoderRNN
from .rnn import *

from base.base_model import BaseModel
from utils.util import *

class ae(BaseModel):
    def __init__(self, args, device=None):
        super(ae, self).__init__()

        args = dict_to_namedtuple(args)
        self.args = args
        self.device = device
        #  self.logger.debug('model name: %s' % self.__class__.__name__)
        #  self.logger.debug(['args in model: ', self.args])

        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)

        self.encoder = rnn_encoder(args, embed=self.embedding)
        #  self.encoder = stack_encoder(args, embed=self.embedding, device=device)

        self.decoder = DecoderRNN(vocab_size = args.vocab_size,
                                  max_len = args.max_len,
                                  hidden_size = args.hidden_size*2 if args.bidirectional else args.hidden_size,
                                  sos_id = args.sos_id,
                                  eos_id = args.eos_id,
                                  n_layers = 1,
                                  rnn_cell = args.rnn_cell,
                                  bidirectional = args.bidirectional, # indicate the diretcions of encoder
                                  input_dropout_p = args.input_dropout_p,
                                  dropout_p = args.dropout_p,
                                  use_attention = args.use_attention,
                                  embed = self.embedding,
                                  args = args)


    def forward(self, x, target, lengths, teacher_forcing_ratio):
        """ @x: (B, L).
        """

        enc_out, enc_h_n = self.encoder(x, lengths)
        #  logging.debug(['the size of enc_h_n: ', enc_h_n.size()])
        #  logging.debug(['the size of encoder_outputs: ', encoder_outputs.size()])
        #  enc_h_n = enc_h_n.transpose(1,2)
        #  enc_h_n = torch.cat(enc_h_n, 0).contiguous().unsqueeze(0).transpose(1,2)

        # dec_outputs:(seq_len, B, vocab_size) the probability of every word
        dec_outputs, dec_hidden, ret_dict = self.decoder(inputs = target,
                                                        encoder_hidden = enc_h_n,
                                                        encoder_outputs = enc_out,
                                                        teacher_forcing_ratio = teacher_forcing_ratio)

        predicts = ret_dict[self.decoder.KEY_SEQUENCE]
        predicts = torch.cat(predicts, 1).contiguous().data.cpu()

        return dec_outputs, predicts


    #  def compute_loss(self, dec_outputs, labels):
    #      """ @dec_outputs:(B, seq_len, vocab_size) the probability of every word
    #          @labels: (B, seq_len). every line represent one document.
    #      """
    #      #labels = labels[:,:-1]
    #      #print(len(dec_outputs))
    #      #print(dec_outputs[0].size())
    #      logits = torch.cat(dec_outputs, 0)#(batch*seq_len, zh_voc)
    #      #print(logits.size())
    #      #logits = logits.contiguous().view(-1, logits.size(-1))
    #      labels = labels.transpose(0,1).contiguous().view(-1)
    #      #  labels = labels.contiguous().view(-1)
    #
    #      loss = torch.mean(self.cost_func(logits, labels))
    #
    #      return loss


