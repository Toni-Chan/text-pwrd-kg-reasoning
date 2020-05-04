'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-05-03 23:36:04
@LastEditors: Please set LastEditors
@Description: cosine similarity function example
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .SimRegulated import SimilarityRegulated
import torch
import torch.nn as nn

class APSynRegulated(SimilarityRegulated):
	def __init__(self, source_embedding=None, entity_dim=50, word_dim=200, **args):
		super(APSynRegulated, self).__init__(**args)
		self.source_embedding = torch.load(source_embedding).cuda()
		self.shrinker = torch.nn.Linear(in_features=word_dim,out_features=entity_dim).cuda()
		self.sim = APSynPower(20,0.1,entity_dim)

	def sim_function(self, data):
		h,t,h_emb,t_emb = self._get_regularization_source(data)
		emb_dist = h_emb - t_emb
		h_wd_emb = self.source_embedding[h]
		t_wd_emb = self.source_embedding[t]
		
		wd_emb_dist = h_wd_emb - t_wd_emb
		shrunken_dist = self.shrinker(wd_emb_dist)
		return self.sim(emb_dist, shrunken_dist)
		

class APSynPower(nn.Module):
	def __init__(self, top_features=20, power=0.1, embed_size=200):
		super(APSynPower, self).__init__()
		self.top_features = top_features
		self.power = power
		self.embed_size = embed_size

	def forward(self, emb_row, wrd_row):
		"""
		APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
		"""

		batch_size = emb_row.shape[0]

		emb_dist_top = torch.argsort(emb_row,dim=1)[:,:self.top_features]
		wrd_dist_top = torch.argsort(wrd_row,dim=1)[:,:self.top_features]

		emb_dist_arr = torch.zeros(batch_size, self.embed_size, dtype=torch.bool).cuda()
		wrd_dist_arr = torch.zeros(batch_size, self.embed_size, dtype=torch.bool).cuda()
		
		emb_dist_arr[emb_dist_top] = True
		wrd_dist_arr[wrd_dist_top] = True

		intersected = ((emb_dist_arr == wrd_dist_arr) * (emb_dist_arr == True)).nonzero()
		score = (2.0 / (torch.pow(emb_row[:,intersected], self.power) + torch.pow(wrd_row[:,intersected], self.power))).sum()
		return score