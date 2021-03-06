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
    def __init__(self, args, eval_model=None, embed = None, device=None):
        super(RL_AE, self).__init__()

        args = dict_to_namedtuple(args)
        self.args = args
        self.device = device

        if embed is not None:
            embed = torch.FloatTensor(embed)
            self.embedding = nn.Embedding.from_pretrained(embed, freeze = True)
            logging.info('using pretrained embedding')
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)

        self.stack_encoder = stack_encoder(args, embed=self.embedding, device=device)

        self.dec_hidden_size = args.hidden_size*2 if args.bidirectional else args.hidden_size
        self.pn_decoder = pn_decoder(vocab_size = args.vocab_size,
                                    hidden_size = self.dec_hidden_size,
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
                                    eval_model = None,
                                    device = device)

        self.eval_model = eval_model # TODO eval_model are auto encoder 
        self.dec_input0 = Parameter(torch.FloatTensor(self.dec_hidden_size), requires_grad=False)
        self.sample_num = args.sample_num

    def forward(self, docs_features, doc_lens, summaries_features, summaries_lens, labels, labels_len, tfr, select_mode='max'):

        # enc_out: for attention, enc_hidden_t: final hidden state (representation), sents_embed: like word embedding
        enc_out, enc_hidden_t, sents_embed = self.stack_encoder(docs_features, doc_lens) # sents_embed:(B, max(doc_lens), 2H)

        #  sents_embed2 = torch.cat([ torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(sents_embed, labels) ]) # (B, labels_max_len, 2H)

        dec_input0 = self.dec_input0.unsqueeze(0).expand(sents_embed.size(0), -1)
        logging.debug(['dec_input0(expected B, 2H[8]): ', dec_input0.size()])
        multi_indices = []
        multi_probs = []
        if self.sample_num > 0:
            for i in range(self.sample_num):
                _, selected_probs, pointers,_ = self.pn_decoder(inputs = sents_embed, 
                                                                decoder_input = dec_input0,
                                                                hidden = enc_hidden_t,
                                                                context = enc_out,
                                                                docs_lens = doc_lens,
                                                                select_mode = select_mode)
                multi_indices.append(pointers)
                multi_probs.append(selected_probs.unsqueeze(0))
            multi_probs = torch.cat(multi_probs) # (sample_num, B, max_dec_len) 

        att_probs, selected_probs, pointers, hidden = self.pn_decoder(inputs = sents_embed, 
                                                        decoder_input = dec_input0,
                                                        hidden = enc_hidden_t,
                                                        context = enc_out,
                                                        docs_lens = doc_lens,
                                                        select_mode = 'max')


        return att_probs, selected_probs, pointers, multi_indices, multi_probs



