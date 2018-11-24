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
        path2save = '%s_%s_%d.pt' % (self.args.path2save, self.model_name, self.save_count)
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
        pad_dim = words_out.size(1) 
        max_doc_len = max(doc_lens)

        sent_input = []
        start = 0
        for doc_len in doc_lens:
            end = start + doc_lens
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

