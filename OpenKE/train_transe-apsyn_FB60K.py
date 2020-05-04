'''
@Author: your name
@Date: 2020-05-01 21:07:37
@LastEditTime: 2020-05-03 23:36:51
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /final/text-pwrd-kg-reasoning/OpenKE-OpenKE-PyTorch/train_transe_FB15K237.py
'''
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import APSynRegulated
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB60K/", 
	nbatches = 1000,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB60K/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = APSynRegulated(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size(),
	entity_dim = 200,
	source_embedding = "benchmarks/FB60K/ref-embeddings.pt",
	l3_regul_rate = 0, 
	sim_regul_rate = 0.01
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 320, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)