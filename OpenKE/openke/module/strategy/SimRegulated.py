'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-04-29 17:10:14
@LastEditors: Please set LastEditors
@Description: Public class of strategy using similarity function
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .Strategy import Strategy

class SimilarityRegulated(Strategy):
	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, sim_regul_rate = 0.0):
		super(SimilarityRegulated, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.sim_regul_rate = sim_regul_rate
        self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score
        
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
		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
        if self.sim_regul_rate != 0:
            loss_res += self.sim_regul_rate * self.sim_function(data)

        return loss_res