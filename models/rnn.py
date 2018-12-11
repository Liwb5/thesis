# coding:utf-8
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .BasicModule import BasicModule
import torch.nn.utils.rnn as rnn_utils


class std_encoder(nn.Module):
    def __init__(self, args, vocab_size, embed=None):
        super(std_encoder, self).__init__()

        #  self.embedding = nn.Embedding(vocab_size, args.embed_dim, padding_idx=0)

        if embed is not None:
            embed = torch.FloatTensor(embed)
            #  self.embedding.weight.data.copy_(embed)
            self.embedding = nn.Embedding.from_pretrained(embed, freeze = True)
            #  self.embedding.weight.requires_grad=True
        else:
            self.embedding = nn.Embedding(vocab_size, args.hidden_size, padding_idx = 0)

        self.args = args
        self.use_cuda = False 
        if args.device is not None:
            self.use_cuda = True

        D = args.embed_dim
        H = args.hidden_size

        self.RNN = nn.GRU(
                        input_size = D, 
                        hidden_size = H, 
                        batch_first = False,
                        bidirectional = True)

    def avg_pool1d(self, x, seq_lens):
        """ @x: (N, L, H).  average pooling in second dimension (L)
            @seq_lens: (N, 1)  
            @output: (N, H)
        """
        output = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :] # (L_n, H)
            t = torch.t(t).unsqueeze(0) # (1, H, L_n)
            output.append(F.avg_pool1d(t, t.size(2)))

        output = torch.cat(output).squeeze(2)  
        return output    # (N, H)

    def order(self, inputs, entext_len):
        """
        order函数将句子的长度按从大到小排序
        inputs: B*en_maxLen. a tensor object
        entext_len: B * 1. the real length of every sentence
        
        return:
        inputs: B * maxLen  tensor
        entext_len: B * 1  tensor
        order_ids:  B * 1  tensor
        """
        #将entext_len按从大到小排序
        sorted_len, sort_ids = torch.sort(entext_len, dim = 0, descending=True)
        
        sort_ids = Variable(sort_ids).cuda() if self.use_cuda else Variable(sort_ids)
        
        inputs = inputs.index_select(0, sort_ids)
        
        _, true_order_ids = torch.sort(sort_ids, 0, descending=False)
        
        #true_order_ids = Variable(true_order_ids).cuda() if self.use_cuda else Variable(true_order_ids)
        
        #排序之后，inputs按照句子长度从大到小排列
        #true_order_ids是原来batch的顺序，因为后面需要将顺序调回来
        return inputs, sorted_len, true_order_ids
 
    def forward(self, x):
        """ @x: (B, seq_len). 
        """
        sent_lens = torch.sum(torch.sign(x), dim=1) # (N, 1). the real length of every sentence
        x, sorted_len, true_order_ids = self.order(x, sent_lens)
        #  logging.debug(['sents_len: ', sent_lens])
        #  logging.debug(['sorted_len: ', sorted_len])
        x = self.embedding(x).transpose(0, 1)    # (L, B, E). E: embedding dimension
        #  logging.debug(['input x size: ', x.size()])
        #  logging.debug(['input x : ', x])
        packed = rnn_utils.pack_padded_sequence(x,lengths = list(sorted_len))

        enc_out, h_n = self.RNN(packed)   # output: (L, B, 2*H) h_n: (num_layer*num_direction, B, H)
        enc_out, _ = rnn_utils.pad_packed_sequence(enc_out)
        enc_out = enc_out.index_select(0, true_order_ids)
        h_n = h_n.index_select(1, true_order_ids)

        #  enc_out = self.avg_pool1d(enc_out, sent_lens)  # output: (N, 2*H)  sentence representation

        return enc_out.transpose(0, 1), h_n
        #  return None, None

# this class has some bugs
class gru_encoder(BasicModule):
    def __init__(self, args, vocab_size, embed=None):
        super(gru_encoder, self).__init__(args)

        if embed is not None:
            self.embedding = embed
        else:
            self.embedding = nn.Embedding(vocab_size, args.embed_dim)

        self.args = args


        D = args.embed_dim
        H = args.hidden_size

        self.word_RNN = nn.GRU(
                        input_size = D, 
                        hidden_size = H, 
                        batch_first = True,
                        bidirectional = True)

        self.sent_RNN = nn.GRU(
                        input_size = 2*H, 
                        hidden_size = H, 
                        batch_first = True,
                        bidirectional = True)

    def avg_pool1d(self, x, seq_lens):
        """ @x: (N, L, H).  average pooling in second dimension (L)
            @seq_lens: (N, 1)  
            @output: (N, H)
        """
        output = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :] # (L_n, H)
            t = torch.t(t).unsqueeze(0) # (1, H, L_n)
            output.append(F.avg_pool1d(t, t.size(2)))

        output = torch.cat(output).squeeze(2)  
        return output    # (N, H)
 
    def forward(self, x, doc_lens):
        """ @x: (N, L). N: sentence number; L: sentence length
            @doc_lens: (B, 1). B: batch size (document number)
        """
        sent_lens = torch.sum(torch.sign(x), dim=1).data # (N, 1). the real length of every sentence
        x = self.embedding(x)    # (N, L, E). E: embedding dimension

        ### word level encode
        H = self.args.hidden_size
        words_out = self.word_RNN(x)[0]   # output: (N, L, 2*H)
        words_out = self.avg_pool1d(words_out, sent_lens)  # output: (N, 2*H)  sentence representation

        # make sent features (pad with zeros)
        x = self.pad_doc(words_out, doc_lens)   # (B, max_doc_len, 2*H)

        ### sentence level encode
        sent_out = self.sent_RNN(x)[0]      # (B, max_doc_len, 2*H)

        docs = self.avg_pool1d(sent_out, doc_lens)   # (B, 2*H) document representation

        return docs




