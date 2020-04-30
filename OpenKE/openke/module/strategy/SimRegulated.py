'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-04-29 17:10:14
@LastEditors: Please set LastEditors
@Description: Public class of strategy using similarity function
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .NegativeSampling import NegativeSampling


class SimilarityRegulated(NegativeSampling):
	def __init__(self, sim_regul_rate=0.0, **args):
		super(SimilarityRegulated, self).__init__(**args)
		self.sim_regul_rate = sim_regul_rate

	def _get_regularization_source(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		h = self.model.ent_embeddings(batch_h)
		t = self.model.ent_embeddings(batch_t)
		return batch_h, batch_t, h, t

	def sim_function(self, data):
		h_emb, t_emb, h, t = self._get_regularization_source(data)
		raise NotImplementedError

	def forward(self, data):
		loss_res = super().forward(data)
		if self.sim_regul_rate != 0:
			loss_res += self.sim_regul_rate * self.sim_function(data)

		return loss_res
