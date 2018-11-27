# coding:utf-8
from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRU_RuNNer(BasicModule):
    def __init__(self, args, embed=None):
        super(GRU_RuNNer, self).__init__(args)
        self.model_name = 'GRU_RuNNer'
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)

        if embed is not None:
            embed = torch.Tensor(embed)
            self.embed.weight.data.copy_(embed)

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

        self.fc = nn.Linear(2*H, 2*H)

        # parameters of classification layer
        self.content = nn.Linear(2*H, 1, bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))


    def max_pool1d(self, x, seq_lens):
        """ @x: (N, L, H).  max_pooling in second dimension (L)
            @seq_lens: (N, 1)  
            @output: (N, H)
        """
        output = []
        for index, t in enumerate(x):
            t = t[:seq_lens[index], :] # (L_n, H)
            t = torch.t(t).unsqueeze(0) # (1, H, L_n)
            output.append(F.max_pool1d(t, t.size(2)))

        output = torch.cat(output).squeeze(2)  
        return output    # (N, H)

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
 
    def word_level_encode(self, words_in):
        """ @words_in: (N, L, E). N: sentence number; L: sentence length; E: embedding dimension
        """
        pass

    def forward(self, x, doc_lens):
        """ @x: (N, L). N: sentence number; L: sentence length
            @doc_lens: (B, 1). B: batch size (document number)
        """
        sent_lens = torch.sum(torch.sign(x), dim=1).data # (N, 1). the real length of every sentence
        x = self.embed(x)    # (N, L, E). E: embedding dimension

        ### word level encode
        H = self.args.hidden_size
        words_out = self.word_RNN(x)[0]   # output: (N, L, 2*H)
        words_out = self.max_pool1d(words_out, sent_lens)  # output: (N, 2*H) 

        # make sent features (pad with zeros)
        x = self.pad_doc(words_out, doc_lens)   # (B, max_doc_len, 2*H)

        ### sentence level encode
        sent_out = self.sent_RNN(x)[0]

        docs = self.max_pool1d(sent_out, doc_lens)

        probs = []
        for index,doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
