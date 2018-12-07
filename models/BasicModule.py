# coding:utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(torch.nn.Module):
    
    def __init__(self, args):
        super(BasicModule, self).__init__()
        self.args = args
        self.model_name = str(type(self))
        self.save_count = 0


    def save(self):
        self.save_count += 1
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        path2save = '%s%s_%d.pt' % (self.args.save_dir, self.model_name, self.save_count)
        torch.save(checkpoint, path2save)

    def load(self, path2load):
        if self.args.device is not None:
            data = torch.load(path2load)['model']
        else:
            data = torch.load(path2load, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self

    def pad_doc(self, words_out, doc_lens):
        """ @words_out: (N, H). N: sentence number; H: hidden size
            @doc_lens: (B, 1). B: batch_size (document number)
        """
        if not isinstance(doc_lens, list):
            doc_lens = [doc_lens]
        pad_dim = words_out.size(1) 
        max_doc_len = max(doc_lens)

        sent_input = []
        start = 0
        for doc_len in doc_lens:
            end = start + doc_len
            valid = words_out[start:end] # output: (doc_len, H)
            start = end

            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))
            else:
                pad = Variable(torch.zeros(max_doc_len - doc_len, pad_dim))
                if self.args.device is not None:
                    pad = pad.cuda()

                sent_input.append(torch.cat([valid, pad]).unsqueeze(0))  # (1, max_doc_len, H)
        
        sent_input = torch.cat(sent_input, dim=0)
        return sent_input   # (B, max_doc_len, H)


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
