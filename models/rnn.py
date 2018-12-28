# coding:utf-8
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn import Parameter
from .attention import Attention 


class rnn_encoder(nn.Module):
    def __init__(self, args, embed=None):
        super(rnn_encoder, self).__init__()

        if embed is not None:
            self.embedding = embed
            #  embed = torch.FloatTensor(embed)
            #  self.embedding.weight.data.copy_(embed)
            #  self.embedding = nn.Embedding.from_pretrained(embed, freeze = True)
            #  self.embedding.weight.requires_grad=True
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_dim) 

        self.args = args

        self.RNN = nn.GRU(input_size = args.embed_dim, 
                        hidden_size = args.hidden_size, 
                        batch_first = False,
                        bidirectional = args.bidirectional)

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

    def forward(self, x, lengths):
        """ @x: (B, seq_len). 
        """
        #  sent_lens = torch.sum(torch.sign(x), dim=1) # (N, 1). the real length of every sentence
        x = self.embedding(x).transpose(0, 1)    # (L, B, E). E: embedding dimension
        packed = rnn_utils.pack_padded_sequence(x,lengths = list(lengths))

        enc_out, h_n = self.RNN(packed)   # output: (L, B, 2*H) h_n: (num_layer*num_direction, B, H)
        enc_out, _ = rnn_utils.pad_packed_sequence(enc_out)

        return enc_out.transpose(0, 1), h_n

class stack_encoder(nn.Module):
    def __init__(self, args, embed=None):
        super(stack_encoder, self).__init__()

        if embed is not None:
            self.embedding = embed
            #  embed = torch.FloatTensor(embed)
            #  self.embedding.weight.data.copy_(embed)
            #  self.embedding = nn.Embedding.from_pretrained(embed, freeze = True)
            #  self.embedding.weight.requires_grad=True
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_dim) 

        self.args = args

        self.word_RNN = nn.GRU(
                        input_size = args.embed_dim,
                        hidden_size = args.hidden_size,
                        batch_first = True,
                        bidirectional = args.bidirectional
                        )
        self.sent_RNN = nn.GRU(
                        input_size = args.hidden_size*2 if args.bidirectional else args.hidden_size,
                        hidden_size = args.hidden_size,
                        batch_first = True,
                        bidirectional = args.bidirectional
                        )
    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out

    def pad_doc(self,words_out,doc_lens):
        if not isinstance(doc_lens, list):
            doc_lens = [doc_lens]

        pad_dim = words_out.size(1)
        max_doc_len = max(doc_lens)
        sent_input = []
        start = 0
        for doc_len in doc_lens:
            stop = start + doc_len
            valid = words_out[start:stop]                                       # (doc_len,2*H)
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))
            else:
                pad = Variable(torch.zeros(max_doc_len-doc_len,pad_dim))
                if self.args.device is not None:
                    pad = pad.cuda()
                sent_input.append(torch.cat([valid,pad]).unsqueeze(0))          # (1,max_len,2*H)
        sent_input = torch.cat(sent_input,dim=0)                                # (B,max_len,2*H)
        return sent_input
 
    def forward(self, x, doc_lens):
        """
        Args:
            x(variables):
            doc_lens(list): (B, 1)
        """
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embedding(x) # batch_size first 

        # word level RNN
        x = self.word_RNN(x)[0]  # (N, sent_lens, 2H)
        word_out = self.max_pool1d(x, sent_lens) #(N, 2H)
        x = self.pad_doc(word_out, doc_lens)
        sent_out = self.sent_RNN(x)[0]
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)

        # sent_out: (B, seq_len, 2H)
        return sent_out, docs, x  # x 是每个句子的表示，用于decoder的时候索引

def pn_decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """
    def __init__(self, vocab_size, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False,
            embed=None, args = None, eval_model=None, max_dec_len=3):

        self.bidirectional_encoder = bidirectional

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.eval_model = eval_model
        self.args = args
        self.max_dec_len = max_dec_len

        self.init_input = None

        self.gru_cell = nn.GRUCell(input_size = hidden_size, 
                                    hidden_size = hidden_size)
        #  if rnn_cell.lower() == 'lstm':
        #      self.rnn_cell = nn.LSTM
        #  elif rnn_cell.lower() == 'gru':
        #      self.rnn_cell = nn.GRU
        #  else:
        #      raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        #  if embed is not None:
        #      self.embedding = embed
        #  else:
        #      self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        if use_attention:
            self.att = Attention(self.hidden_size, self.hidden_size)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def _init_mask(lens):
        L = lens.data.view(-1).tolist
        batch_size = len(L)
        mask = self.mask.repeat(max(L)).unsqueeze(0).repeat(batch_size, 1)
        # mask padding index
        for i in range(batch_size):
            mask[i][L[i]:] = 0
        return mask

    def forward(self, inputs, decoder_input, hidden, context, docs_lens):
        """
        Args:
            inputs(B, labels_len, hidden_size): sentence embedding, 每一步的decoder的输入从这里选择 
            decoder_input(B, hidden_size): 初始decoder的输入，一般是0 
            hidden(B, hidden_size): 解码器的representation，用于表示整篇文档,作为decoder的初始hidden_state
            context(B, seq_len, hidden_size): 编码器每个step的hidden_state，用于attention
            docs_lens(B, 1): 每篇文档的句子数量, 用于mask padding index
        """
        batch_size = context.size(0)
        input_length = context.size(1)
        
        mask = self._init_mask(docs_lens)  
        self.att.init_inf(mask.size())   

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t h, Attention probabilities (Alpha)
            """
            # GRU_cell
            h_t = self.gru_cell(x, hidden)

            # Attention section
            _, alpha = self.att(h_t, context, torch.eq(mask, 0))
            #  hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            # h_t(B, hidden_size); alpha(B, seq_len)
            return h_t, alpha 

        for _ in range(min(self.max_dec_len, input_length)):
            hidden, outs = step(decoder_input, hidden)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            # TODO e-greedy sample 
            # max_probs: (B, 1) indices: (B, 1) 
            max_probs, indices = masked_outs.max(1)

            # runner 每一行都是从0,1,2,3递增，one_hot_pointers是为了得到当前step所选择的对应位置
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.hidden_size).byte()
            decoder_input = inputs[embedding_mask.data].view(batch_size, self.hidden_size)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  #(B, max_dec_len, seq_len)
        pointers = torch.cat(pointers, 1) # (B, max_dec_len)

        return outputs, pointers, hidden
            
