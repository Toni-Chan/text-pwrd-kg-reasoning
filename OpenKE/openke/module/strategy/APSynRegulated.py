'''
@Author: your name
@Date: 2020-05-03 03:18:17
@LastEditTime: 2020-05-03 19:20:11
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/APSymRegulated.py
'''
'''
@Author: toni_chan
@Date: 2020-04-29 16:58:03
@LastEditTime: 2020-05-03 03:18:17
@LastEditors: Please set LastEditors
@Description: cosine similarity function example
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE/openke/module/strategy/SimRegulated.py
'''
from .SimRegulated import SimilarityRegulated
import torch
import torch.nn as nn

class APSynRegulated(SimilarityRegulated):
	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, sim_regul_rate = 0.0,
				source_embedding=None, entity_dim=50, word_dim=200):
		super(APSynRegulated, self).__init__(model, loss, batch_size, regul_rate, l3_regul_rate, sim_regul_rate)
		self.source_embedding = torch.load(source_embedding)
		self.shrinker = torch.nn.Linear(in_features=word_dim,out_features=entity_dim)
		self.sim = APSynPower()

	def sim_function(self, data):
		h_emb, t_emb, h, t = self._get_regularization_source(data)
		emb_dist = h_emb - t_emb
		h_wd_emb = self.source_embedding[h]
		t_wd_emb = self.source_embedding[t]
		
		wd_emb_dist = h_wd_emb - t_wd_emb
		shrunken_dist = self.shrinker(wd_emb_dist)
		return self.sim(emb_dist, shrunken_dist)
		

class APSynPower(nn.Module):
	def __init__(self):
		super(APSynPower, self).__init__()

	def APSyn(self, x_row, y_row):
		"""
		APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
		:param x_row:
		:param y_row:
		:return: similarity score
		"""

		# Sort y's contexts
		y_contexts_cols = self.sort_by_value_get_col(y_row) # tuples of (row, col, value)
		y_context_rank = { c : i + 1 for i, c in enumerate(y_contexts_cols) }

		# Sort x's contexts
		x_contexts_cols = self.sort_by_value_get_col(x_row)

		assert len(x_contexts_cols) == len(y_contexts_cols)

		x_context_rank = { c : i + 1 for i, c in enumerate(x_contexts_cols) }

		# Average of 1/(rank(w1)+rank(w2)/2) for every intersected feature among the top N contexts
		intersected_context = set(y_contexts_cols).intersection(set(x_contexts_cols))
		
		if formula == F_ORIGINAL:
		score = sum([2.0 / (x_context_rank[c] + y_context_rank[c]) for c in intersected_context]) #Original
		elif formula == F_POWER:
		score = sum([2.0 / (math.pow(x_context_rank[c], POWER) + math.pow(y_context_rank[c], POWER)) for c in intersected_context])
		elif formula == F_BASE_POWER:
		score = sum([math.pow(BASE, (x_context_rank[c]+y_context_rank[c])/2.0) for c in intersected_context])
		else:
		sys.exit('Formula value not found!')

		return score

	def sort_by_value_get_col(self, mat):
		"""
		Sort a sparse coo_matrix by values and returns the columns (the matrix has 1 row)
		:param mat: the matrix
		:return: a sorted list of tuples columns by descending values
		"""

		sorted_tuples = sorted(mat, key=lambda x: x[2], reverse=True)

		if len(sorted_tuples) == 0: return []

		rows, columns, values = zip(*sorted_tuples)
		return columns