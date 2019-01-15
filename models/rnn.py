# coding:utf-8
from pprint import pprint, pformat
import logging
import numpy as np
import random
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
                        batch_first = True,
                        bidirectional = args.bidirectional)

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:] # get rid of padding index
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out

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
        sent_lens = torch.sum(torch.sign(x), dim=1) # (N, 1). the real length of every sentence
        x = self.embedding(x)    # (B, L, E). E: embedding dimension
        #  packed = rnn_utils.pack_padded_sequence(x,lengths = list(lengths))

        word_hidden, _ = self.RNN(x)   # output: (L, B, 2*H) h_n: (num_layer*num_direction, B, H)
        h_n = self.max_pool1d(word_hidden, sent_lens) #(N, 2H)
        #  enc_out, _ = rnn_utils.pad_packed_sequence(enc_out)

        return word_hidden, h_n


class stack_encoder(nn.Module):
    def __init__(self, args, embed=None, device=None):
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
        self.device = device

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
            t = t[:seq_lens[index],:] # get rid of padding index
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))
        
        out = torch.cat(out).squeeze(2)
        return out

    def pad_doc(self,words_out,doc_lens):
        if not isinstance(doc_lens, list):
            doc_lens = [doc_lens]

        pad_dim = words_out.size(1) # 2H
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
                if self.device is not None:
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
        #  logging.debug(['encoder: sent_lens: ', sent_lens])
        word_embed = self.embedding(x) # batch_size first 

        # word level RNN
        word_hidden = self.word_RNN(word_embed)[0]  # (N, sent_lens, 2H)
        #  logging.debug(['after word_RNN(expected N,max_sent_lens[4], 2H[8]): ', word_hidden.size()])
        sent_embed = self.max_pool1d(word_hidden, sent_lens) #(N, 2H)
        #  logging.debug(['after max_pool1d(expected N, 2H[8]): ', sent_embed.size()])
        sent_embed  = self.pad_doc(sent_embed, doc_lens)
        logging.debug(['after pad_doc, sent_embed(expected B,max_doc_len, 2H[8]): ', sent_embed.size()])
        sent_hidden = self.sent_RNN(sent_embed)[0]
        logging.debug(['after sent_RNN, sent_hidden(expected B,max_doc_len, 2H[8]): ', sent_hidden.size()])
        doc_embed = self.max_pool1d(sent_hidden,doc_lens)                                # (B,2*H)
        logging.debug(['after doc max_pool1d, doc_embed(expected B, 2H[8]): ', doc_embed.size()])

        # sent_hidden: (B, max_doc_len, 2H) 是每个句子对应的hidden state
        # doc_embed: (B, 2H) # 每篇文档对应的表示
        # sent_embed(B, max_doc_len, 2H) 是每个句子的表示
        return sent_hidden, doc_embed, sent_embed 

class pn_decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """
    def __init__(self, vocab_size, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False,
            embed=None, args = None, eval_model=None, max_dec_len=3, 
            device=None):

        super(pn_decoder, self).__init__()
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
        self.max_dec_len = args.max_selected
        self.select_mode = args.select_mode
        self.device = device

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

    def _init_mask(self, lens):
        #  lens = lens.data.view(-1).tolist
        batch_size = len(lens)
        mask = self.mask.repeat(max(lens)).unsqueeze(0).repeat(batch_size, 1)
        # mask padding index
        for i in range(batch_size):
            mask[i][lens[i]:] = 0
        return mask

    def forward(self, inputs, decoder_input, hidden, context, docs_lens, epsilon=0):
        """
        Args:
            inputs(B, labels_len, hidden_size): sentence embedding, 每一步的decoder的输入从这里选择 
            decoder_input(B, hidden_size): 初始decoder的输入，一般是0 
            hidden(B, hidden_size): 解码器的representation，用于表示整篇文档,作为decoder的初始hidden_state
            context(B, seq_len, hidden_size): 编码器每个step的hidden_state，用于attention
            docs_lens(B, 1): 每篇文档的句子数量, 用于mask padding index
        """
        batch_size = context.size(0)
        
        mask = self._init_mask(docs_lens)  
        #  logging.debug(['mask size(expected B, max_doc_len[3], 2H[8]): ', mask.size()])
        #  logging.debug(['mask (expected B, max_doc_len[3]): ', pformat(mask.data.cpu().numpy())])
        self.att.init_inf(mask.size())   

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(mask.size(1))
        for i in range(mask.size(1)):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()
        #  logging.debug(['runner (expected B, max_doc_len[3]): ', pformat(runner.data.cpu().numpy())])

        outputs = []
        pointers = []
        selected_probs = []

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

        #  logging.debug(['sent_embed(inputs in decoder) (B, max_doc_len[3], 2H): ', inputs.data.cpu().numpy()])
        for _ in range(min(self.max_dec_len, min(docs_lens))):
            hidden, att_probs = step(decoder_input, hidden)
            logging.debug(['step output att_probs(attention probs)(expected B, max_doc_len[3]): ', att_probs.data.cpu().numpy()])

            # Masking selected inputs
            #  masked_outs = att_probs * mask  # warning: do not use this operation because att_probs maybe negative number in log_softmax in Attention
            #  logging.debug(['masked_outs (expected B, max_doc_len[3]): ', mask.data.cpu().numpy()])

            # Get maximum probabilities and indices
            # TODO e-greedy sample 
            # max_probs: (B, 1) indices: (B, 1) 
            #  max_probs, indices = masked_outs.max(1)
            #  selected_prob, indices = att_probs.max(1)
            selected_prob, indices = self.select_sent_indices(att_probs, mask, epsilon)
            logging.debug(['selected indices (expected B, 1): ', indices.data.cpu().numpy()])

            # runner 每一行都是从0,1,2,3递增，one_hot_pointers是为了得到当前step所选择的对应位置
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, att_probs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.hidden_size).byte()
            decoder_input = inputs[embedding_mask.data].view(batch_size, self.hidden_size)
            #  logging.debug(['decoder input in every step(B, 2H): ', decoder_input.data.cpu().numpy()])

            outputs.append(att_probs.unsqueeze(0))
            selected_probs.append(selected_prob.unsqueeze(1))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)  #(B, max_dec_len, seq_len)
        pointers = torch.cat(pointers, 1) # (B, max_dec_len)
        selected_probs = torch.cat(selected_probs, 1) # (B, max_dec_len)
        logging.debug(['all att_probs (B, min_doc_lens, max_doc_len): ', outputs.size()])
        #  logging.debug(['all att_probs (B, min_doc_lens, max_doc_len): ', outputs.data.cpu().numpy()])
        logging.debug(['pointers (B, min_doc_lens): ', pointers.size()])
        #  logging.debug(['pointers (B, min_doc_lens): ', pointers.data.cpu().numpy()])
        #  logging.debug(['selected_probs (B, min_doc_lens): ', selected_probs.data.cpu().numpy()])

        return outputs, selected_probs, pointers, hidden

    def select_sent_indices(self, att_probs, mask, epsilon):
        greedy = False if random.random() < epsilon else True 
        if greedy:
            selected_prob, indices = att_probs.max(1)

        elif self.select_mode == 'random':  # random sample index
            #  logging.debug('in random')
            mask_arr = mask.data.cpu().numpy()
            indices = []
            for i in range(mask_arr.shape[0]):
                indices.append(np.random.choice(np.where(mask_arr[i,:] == 1)[0]))
            indices = torch.LongTensor(indices)
            if self.device is not None:
                indices = indices.cuda()
            selected_prob = att_probs.gather(1, indices.view(-1,1)).squeeze(1)

        elif self.select_mode == 'distribute': # sample index with distribution
            #  logging.debug('in distribute')
            att_probs_arr = att_probs.data.cpu().numpy()
            length = att_probs_arr.shape[1]
            indices = []
            for i in range(att_probs_arr.shape[0]):
                indices.append(np.random.choice(length, p = att_probs_arr[i,:]))
            indices = torch.LongTensor(indices)
            if self.device is not None:
                indices = indices.cuda()
            selected_prob = att_probs.gather(1, indices.view(-1,1)).squeeze(1)
        else:
            selected_prob, indices = att_probs.max(1)

        return selected_prob, indices

class rnn_decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False,
            embed=None, args = None, eval_model=None, max_dec_len=3):

        super(pn_decoder, self).__init__()
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

        self.gru_cell = nn.GRUCell(input_size = hidden_size, 
                                    hidden_size = hidden_size)

        if use_attention:
            self.att = Attention(self.hidden_size, self.hidden_size)

    def forward(self, inputs, decoder_input, hidden, context, teacher_forcing_ratio=0):
        """
        Args:
            inputs(B, labels_len, hidden_size): sentence embedding, 每一步的decoder的输入从这里选择 
            decoder_input(B, hidden_size): 初始decoder的输入，一般是0 
            hidden(B, hidden_size): 解码器的representation，用于表示整篇文档,作为decoder的初始hidden_state
            context(B, seq_len, hidden_size): 编码器每个step的hidden_state，用于attention
        """

        use_tfr = True if random.random() < teacher_forcing_ratio else False

        batch_size = context.size(0)


