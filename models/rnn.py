# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .BasicModule import BasicModule


class std_encoder(nn.Module):
    def __init__(self, args, vocab_size, embed=None):
        super(std_encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, args.embed_dim, padding_idx=0)

        if embed is not None:
            embed = torch.FloatTensor(embed)
            #  self.embedding.weight.data.copy_(embed)
            self.embedding = nn.Embedding.from_pretrained(embed)
            #  self.embedding.weight.requires_grad=True
        else:
            self.embedding = nn.Embedding(vocab_size, args.hidden_size, padding_idx = 0)

        self.args = args

        D = args.embed_dim
        H = args.hidden_size

        self.RNN = nn.GRU(
                        input_size = D, 
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
 
    def forward(self, x):
        """ @x: (B, L). 
        """
        sent_lens = torch.sum(torch.sign(x), dim=1).data # (N, 1). the real length of every sentence
        x = self.embedding(x)    # (B, L, E). E: embedding dimension

        H = self.args.hidden_size
        enc_out = self.RNN(x)[0]   # output: (N, L, 2*H)
        enc_out = self.avg_pool1d(enc_out, sent_lens)  # output: (N, 2*H)  sentence representation

        return enc_out 

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




