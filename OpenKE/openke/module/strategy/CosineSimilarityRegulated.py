'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-05-03 19:36:53
@LastEditors: Please set LastEditors
@Description: cosine similarity function example
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .SimRegulated import SimilarityRegulated
import torch
import torch.nn as nn

class CosineSimilarityRegulated(SimilarityRegulated):
	def __init__(self, source_embedding=None, entity_dim=50, word_dim=200, **args):
		super(CosineSimilarityRegulated, self).__init__(**args)
		self.source_embedding = torch.load(source_embedding).cuda()
		self.shrinker = nn.Linear(in_features=word_dim,out_features=entity_dim).cuda()
		self.sim = nn.CosineSimilarity(dim=1)
		self.entity_dim = entity_dim
		self.word_dim = word_dim

	def sim_function(self, data):
		h,t,h_emb,t_emb = self._get_regularization_source(data)
		
		emb_dist = h_emb - t_emb
		h_wd_emb = self.source_embedding[h]
		t_wd_emb = self.source_embedding[t]
		
		wd_emb_dist = h_wd_emb - t_wd_emb
		shrunken_dist = self.shrinker(wd_emb_dist)
		result = self.sim(emb_dist, shrunken_dist)
		bs = h.shape[0] // 2
		mask = torch.zeros(bs * 2,1).cuda()
		mask[:bs,:] = 1
		return (result * mask).mean()
		

		
		
		
		