import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class ConvE(Model):
    def __init__(self, ent_tot, rel_tot, hidden, embedding_shape, \
            dim=100, bias_flag=True, drops=[0,0,0]):
        super(ConvE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.ip_drop = drops[0]
        self.hid_drop = drops[1]
        self.feat_drop = drops[2]
        self.emb_dim1 = embedding_shape
        self.emb_dim2 = dim // self.emb_dim1

        self.ent_embeddings = torch.nn.Embedding(ent_tot, dim, padding_idx=0)
        self.rel_embeddings = torch.nn.Embedding(rel_tot, dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(self.ip_drop)
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.feat_drop)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=bias_flag)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.fc = torch.nn.Linear(hidden, dim)

        self.register_parameter('b', torch.nn.Parameter(torch.zeros(ent_tot)))

        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.xavier_normal_(self.rel_embeddings.weight.data)

    def _calc(self, h, r, t):
        h_conv = h.view(-1, 1, self.emb_dim1, self.emb_dim2)
        r_conv = r.view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat((h_conv, r_conv), 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        #x = torch.mm(x, self.ent_embeddings.weight.transpose(1,0))
        #x += self.b.expand_as(x)
        score = torch.sum(x * t,-1)
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        if data['mode'] == "head_batch" and batch_h.shape[0] != batch_r.shape[0]:
            batch_r = batch_r.repeat(batch_h.shape[0])
            batch_t = batch_t.repeat(batch_h.shape[0])

        if data['mode'] == "tail_batch" and batch_t.shape[0] != batch_r.shape[0]:
            batch_r = batch_r.repeat(batch_t.shape[0])
            batch_h = batch_h.repeat(batch_t.shape[0])

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
            

        score = self._calc(h,r,t)
        return score

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data): # regularization should be useless.
		# batch_h = data['batch_h']
		# batch_t = data['batch_t']
		# batch_r = data['batch_r']
		# h = self.ent_embeddings(batch_h)
		# t = self.ent_embeddings(batch_t)
		# r = self.rel_embeddings(batch_r)
		# regul = (torch.mean(h ** 2) + 
		# 		 torch.mean(t ** 2) + 
		# 		 torch.mean(r ** 2)) / 3
        return 0