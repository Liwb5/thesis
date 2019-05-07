import models.inferSent as module_encoder
from utils.config import *
import torch
import torch.nn as nn
import numpy as np

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

config = get_config_from_yaml('./configs/config.yaml')
config = process_config(config)

inferSent = getattr(module_encoder, config['inferSent']['type'])(config['inferSent']['args'])
inferSent.load_state_dict(torch.load(config['inferSent']['model_path']))
inferSent.load_vocab(config['inferSent']['w2v_path'])
inferSent.cuda()

task_name = 'train.inferSentReward_new_loss_hs200_embed_300d_file_rl_advantage_R_1Rewards_explorate_rate3_lr0.0000001.v2_5.big.0312'
number = 3
total_article = 10350

ref_path = './outputs/%s/ref%d/' %(task_name, number)
hyp_path = './outputs/%s/hyp%d/' %(task_name, number)

score = []
for i in range(total_article): # total test article
    with open(ref_path+'ref.A.%05d.txt'%(i), 'r') as f:
        ref_sum = f.read()
        ref_sum = ' '.join(ref_sum.split('\n'))
    with open(hyp_path+'hyp.%05d.txt'%(i), 'r') as f:
        hyp_sum = f.read()
        hyp_sum = ' '.join(hyp_sum.split('\n'))
        
    score.append(cosine(inferSent.encode(ref_sum)[0], inferSent.encode(hyp_sum)[0]))
    if i % 200 == 0:
        print('have processed %d'%i)
    
avg_score = sum(score)/total_article

print(score)
print('average score: ', avg_score)
