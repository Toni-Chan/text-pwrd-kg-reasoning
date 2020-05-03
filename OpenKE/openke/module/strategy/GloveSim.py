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
        rows, cols = self.weight_matrix.nonzero()
        pos = zip(rows, cols)
        head = data['batch_h']
        tail = data['batch_t']

        # data(69628, )(69628, ) head.shape,tail.shape
        ## add similarity regularization term
        head_batch = []
        tail_batch = []
        head_sim = []
        tail_sim = []
        for item in zip(head, tail):
            val1 = item[0].tolist()
            val2 = item[1].tolist()
            if (min(val1, val2), max(val1, val2)) in pos:
                head_batch.append(val1)
                tail_batch.append(val2)
                head_sim.append(min(val1, val2))
                tail_sim.append(max(val1, val2))
        if not len(head_sim):
            return  torch.FloatTensor([0]).cuda()
        ## calculate wi wj
        head_batch = Variable(torch.LongTensor(head_batch).cuda())
        tail_batch = Variable(torch.LongTensor(tail_batch).cuda())
        embed_weight = self.model.ent_embeddings.weight
        head_weight = embed_weight[head_batch]
        tail_weight = embed_weight[tail_batch]
        batch_weight = torch.bmm(head_weight.unsqueeze(1), tail_weight.unsqueeze(2)).view(-1) # batch_size, 1, 200,)(batch_size, 200, 1)
        ## calculate X ij
        sim_weight = Variable(torch.from_numpy(self.weight_matrix[head_sim, tail_sim])).view(-1)
        print("weight",self.weight_matrix[head_sim, tail_sim])
        sim_weight = sim_weight.type(torch.FloatTensor).cuda()
        regularized_loss = torch.sum((batch_weight - torch.log(sim_weight)) ** 2).cuda()

        return regularized_loss