import pickle
import numpy as np
import bcolz
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling, SimilarityRegulated
from openke.data import TrainDataLoader, TestDataLoader
import json
import scipy

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB60K/",
	nbatches = 100,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB60K/", "link")

#load pretrained glove dictionary
glove_path = "/home/ubuntu/text-pwrd-kg-reasoning/data/"
vectors = bcolz.open(f'{glove_path}/6B.200.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.200_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.200_idx.pkl', 'rb'))
# wrd2embedding
glove = {w: vectors[word2idx[w]] for w in words}
entity2name = {}

with open(f'{glove_path}/mid2name.tsv', 'r') as f:
	line = f.readline().split()
	while line:
		entity2name[line[0]] = line[1:]
		line = f.readline().split()

id2entity = {}
with open("/home/ubuntu/text-pwrd-kg-reasoning/OpenKE/benchmarks/FB60K/entity2id.txt") as f:
	line = f.readline().split()
	max_id = int(line[0])
	line = f.readline().split()
	while line:
		id2entity[int(line[1])] = line[0]
		line = f.readline().split()

# create weight matrix for entity
matrix_len = train_dataloader.get_ent_tot()
print("entity size:", matrix_len)
weights_matrix = np.zeros((matrix_len, 200))

for i in range(max_id):
	entity = id2entity[i]
	try:
		words = entity2name[entity]
	except KeyError:
		words = ['unk']
	for wrd in words:
		try:
			weights_matrix[i] += glove[wrd]
		except KeyError:
			weights_matrix[i] += glove['unk']
	weights_matrix[i] /= len(words)

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	ent_weight  = weights_matrix,
	# rel_weight = cur_rel_weight,
	dim = 200,
	p_norm = 1,
	norm_flag = True)


# define the loss function
# model = NegativeSampling(
model = NegativeSampling(
	model = transe,
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)
base_path = '/home/ubuntu/text-pwrd-kg-reasoning/OpenKE/benchmarks/FB60K/'
sim_weight = scipy.sparse.load_npz(base_path + 'sparse_matrix.npz')
# with open(base_path + 'weight.json') as f:
# 	sim_weight = json.load(f)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True, sim_weight=sim_weight)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
