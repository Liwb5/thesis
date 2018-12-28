# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys
import logging
from .rnn import *

from base.base_model import BaseModel
from utils.util import *

class RL_AE(BaseModel):
    def __init__(self, args, eval_model=None):
        super(RL_AE, self).__init__()

        args = dict_name_tuple(args)
        self.args = args

        self.docs_encoder = stack_encoder(args, embed=None)

        self.pn_decoder = pn_decoder(vocab_size = args.vocab_size,
                                    hidden_size = args.hidden_size*2 if args.bidirectional else args.hidden_size,
                                    sos_id = args.sos_id, 
                                    eos_id = args.eos_id,
                                    n_layers = 1,
                                    rnn_cell = args.rnn_cell,
                                    bidirectional = args.bidirectional,
                                    input_dropout_p = args.input_dropout_p,
                                    dropout_p = args.dropout_p,
                                    use_attention = True,
                                    embed = None,
                                    args = args,
                                    eval_model = None)

        self.eval_model = eval_model

    def forward(self, docs_features, doc_lens, summaries_features, summaries_lens, labels):

        enc_out, enc_hidden_t, sents_embed = self.stack_encoder(docs_features, doc_lens) # sents_embed:(B, max(doc_lens), 2H)

        decoder_inputs = torch.cat([ torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sents_embed, labels) ]) # (B, labels_max_len, 2H)

        dec_outputs, pred_index,  hidden = self.pn_decoder(inputs = sents_embed, 
                                                        decoder_input = enc_hidden_t,
                                                        context = enc_out)



