'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-04-29 17:10:14
@LastEditors: Please set LastEditors
@Description: Public class of strategy using similarity function
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .SimRegulated import SimilarityRegulated
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class GloveSimGeneral(SimilarityRegulated):
    def __init__(self, weight_matrix=None, **args):
        super(GloveSimGeneral, self).__init__(**args)
        self.weight_matrix = weight_matrix

    def sim_function(self, data):
        head = data['batch_h'].cpu().detach().numpy()
        tail = data['batch_t'].cpu().detach().numpy()
        ## vectorize:
        weights = np.array(self.weight_matrix[head.tolist(), tail.tolist()]).reshape(-1)
        inds = np.argwhere(weights != 0)
        reduced_sim_weight = weights[inds]
        reduced_sim_weight = Variable(torch.from_numpy(reduced_sim_weight)).view(-1).cuda()
        head = head[inds]
        tail = tail[inds]
        if not len(head):
            return  torch.FloatTensor([0]).cuda()
        ## calculate wi wj
        embed_weight = self.model.ent_embeddings.weight
        head_weight = embed_weight[head]
        tail_weight = embed_weight[tail]
        batch_weight = torch.bmm(head_weight, tail_weight.permute(0,2,1)).view(-1) # batch_size, 1, 200,)(batch_size, 200, 1)
        ## calculate X ij
        # print(batch_weight)
        # print(reduced_sim_weight)
        regularized_loss = torch.sum((batch_weight - reduced_sim_weight) ** 2).cuda()
        # print(regularized_loss)
        return regularized_loss