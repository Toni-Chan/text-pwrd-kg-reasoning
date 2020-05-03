# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm

class Trainer(object):

	def __init__(self, 
				 model = None,
				 data_loader = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None,
				 sim_weight = None
				 ):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data_loader = data_loader
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		self.sim_weight = sim_weight

	def train_one_step(self, data):
		self.optimizer.zero_grad()
		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode']
		})

		### vectorize
		## load sparse matrix
		rows, cols = self.sim_weight.nonzero()
		head = data['batch_h']
		tail = data['batch_t']
		# data(69628, )(69628, ) head.shape,tail.shape
		## add similarity regularization term
		head_batch= []
		tail_batch = []
		head_sim = []
		tail_sim = []
		for item in zip(head,tail):
			val1 = item[0]
			val2 = item[1]
			if min(val1,val2) in rows and max(val1,val2) in cols:
				head_batch.append(val1)
				tail_batch.append(val2)
				head_sim.append(min(val1,val2))
				tail_sim.append(max(val1,val2))

		head_batch = Variable(torch.LongTensor(head_batch).cuda())
		tail_batch = Variable(torch.LongTensor(tail_batch).cuda())
		embed_weight = self.model.model.ent_embeddings.weight
		head_weight = embed_weight[head_batch]
		tail_weight = embed_weight[tail_batch]
		# (batch_size, 1, 200,) (batch_size, 200, 1)
		batch_weight = torch.bmm(head_weight.unsqueeze(1), tail_weight.unsqueeze(2)).view(-1)
		sim_weight = Variable(torch.from_numpy(self.sim_weight[head_sim,tail_sim]).cuda()).view(-1)
		# print("weight size",batch_weight.size(), sim_weight.size())
		regularized_loss = torch.sum((batch_weight - sim_weight) **2)
		## loop
		# for i in range():
		# 	cur_head = embed_weight[[head[i]]]
		# 	cur_tail = embed_weight[tail[i]]
		# 	head_idx = head[i].cpu().detach().numpy()
		# 	tail_idx = tail[i].cpu().detach().numpy()
		# 	fin_key = str(head_idx) + '_' + str(tail_idx) if head_idx < tail_idx else str(tail_idx) + '_' + str(
		# 		head_idx)
		# 	if fin_key in self.sim_weight.keys():
		# 		similarity = torch.Tensor([self.sim_weight[fin_key]]).cuda()
		# 		# print(similarity)
		# 		dist = (torch.dot(cur_head, cur_tail.T) - torch.log(similarity)) ** 2
		# 		regularized_loss.append(dist)
		# regularized_loss = torch.cat(regularized_loss)
		# regularized_loss = torch.sum(regularized_loss)
		loss += regularized_loss
		## backward
		loss.backward()
		self.optimizer.step()
		print("step")
		return loss.item()

	def run(self):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		for epoch in training_range:
			res = 0.0
			for data in self.data_loader:
				loss = self.train_one_step(data)
				res += loss
			training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
			
			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir