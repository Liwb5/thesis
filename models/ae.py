# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import sys
sys.path.append('../')
from seq2seq.models import DecoderRNN
from seq2seq.models import EncoderRNN
from seq2seq.models import TopKDecoder
from .rnn import *
from .BasicModule import BasicModule

class ae(BasicModule):
    def __init__(self, args, embed=None):
        super(ae, self).__init__(args)

        self.args = args
        self.model_name = 'ae'

        #  if embed is not None:
        #      src_embedding = embed['src_emb']
        #      tgt_embedding = embed['tgt_emb']
        #  else:
        #      src_embedding = None
        #      tgt_embedding = None

        self.encoder = std_encoder(args, args.embed_num, embed)

        self.decoder = DecoderRNN(vocab_size = args.embed_num,
                                  max_len = 100,
                                  hidden_size = args.hidden_size*2,
                                  sos_id = 2,
                                  eos_id = 3,
                                  n_layers = 1,
                                  rnn_cell = 'gru',
                                  bidirectional = True, # indicate the diretcions of encoder
                                  input_dropout_p = 0,
                                  dropout_p = 0,
                                  use_attention = False,
                                  embed = embed)

        self.cost_func = nn.CrossEntropyLoss(weight = args.weights)

    def forward(self, x, target):
        """ @x: (B, L). 
        """

        encoder_outputs, enc_h_n = self.encoder(x) 
        #  print('the size of enc_h_n: ', enc_h_n.size())
        #  print('the size of encoder_outputs: ', encoder_outputs.size())
        #  enc_h_n = enc_h_n.transpose(1,2)
        #  enc_h_n = torch.cat(enc_h_n, 0).contiguous().unsqueeze(0).transpose(1,2)

        # dec_outputs:(seq_len, B, vocab_size) the probability of every word
        dec_outputs, dec_hidden, ret_dict = self.decoder(inputs = target,
                                                        encoder_hidden = enc_h_n,
                                                        encoder_outputs = encoder_outputs.transpose(0,1),
                                                        teacher_forcing_ratio = 1.0)

        predicts = ret_dict[self.decoder.KEY_SEQUENCE]
        predicts = torch.cat(predicts, 1).contiguous().data.cpu()

        return dec_outputs, predicts
        #  return None, None


    def compute_loss(self, dec_outputs, labels):
        """ @dec_outputs:(B, seq_len, vocab_size) the probability of every word 
            @labels: (B, seq_len). every line represent one document. 
        """
        #labels = labels[:,:-1]
        #print(len(dec_outputs))  
        #print(dec_outputs[0].size())
        logits = torch.cat(dec_outputs, 0)#(batch*seq_len, zh_voc)
        #print(logits.size())
        #logits = logits.contiguous().view(-1, logits.size(-1))
        #  labels = labels.transpose(0,1).contiguous().view(-1)
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss


if __name__=='__main__':
    print('testing ae.py file')
