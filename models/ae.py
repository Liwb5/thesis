# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import models

from seq2seq.model import DecoderRNN
from seq2seq.model import EncoderRNN
from seq2seq.model import TopKDecoder

class ae(nn.Module):
    def __init__(self, args, embed=None):
        super(ae, self).__init__()

        self.args = args
        self.model_name = 'ae'

        #  if embed is not None:
        #      src_embedding = embed['src_emb']
        #      tgt_embedding = embed['tgt_emb']
        #  else:
        #      src_embedding = None
        #      tgt_embedding = None

        self.encoder = models.gru_encoder(args, args.tgt_vocab_size, embed)

        self.decoder = DecoderRNN(vocab_size = args.tgt_vocab_size,
                                  max_len = 100,
                                  hidden_size = args.hidden_size*2,
                                  sos_id = 1,
                                  eos_id = 2,
                                  n_layers = 1,
                                  rnn_cell = 'lstm',
                                  bidirectional = False,
                                  input_dropout_p = 0,
                                  dropout_p = 0,
                                  use_attention = False)

        self.cost_func = nn.CrossEntropyLoss()

    def forward(self, x, docs_lens):
        """ @x: (N, L). N: sentence number; L: sentence length 
            @doc_lens: (B, 1). B: batch size (document number)
        """

        encoder_outputs = self.encoder(x, doc_lens) # output: (B, 2*H)

        # dec_outputs:(seq_len, B, vocab_size) the probability of every word
        dec_outputs, dec_hidden, ret_dict = self.decoder(inputs = labels,
                                                        encoder_outputs = encoder_outputs,
                                                        teacher_forcing_ratio = 1.0)

        predicts = ret_dict[self.decoder.KEY_SEQUENCE]
        predicts = torch.cat(predicts, 1).contigous().data.cpu()

        return dec_outputs, predicts


    def compute_loss(self, dec_outputs, labels):
        """ @dec_outputs:(B, seq_len, vocab_size) the probability of every word 
            @labels: (B, seq_len). every line represent one document. 
        """
        #labels = labels[:,:-1]
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
        else:
            labels = Variable(labels).long()
        #print(len(dec_outputs))  
        #print(dec_outputs[0].size())
        logits = torch.cat(dec_outputs, 0)#(batch*seq_len, zh_voc)
        #print(logits.size())
        #logits = logits.contiguous().view(-1, logits.size(-1))
        #  labels = labels.transpose(0,1).contiguous().view(-1)
        labels = torch.cat(labels.contiguous(), 0)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss

