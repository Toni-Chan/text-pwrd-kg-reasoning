from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling, SimilarityRegulated, GloveSimGeneral
from openke.data import TrainDataLoader, TestDataLoader
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

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200,
	p_norm = 1,
	norm_flag = True)

base_path = '/home/ubuntu/text-pwrd-kg-reasoning/OpenKE/benchmarks/FB60K/'
sim_weight = scipy.sparse.load_npz(base_path + 'sparse_matrix.npz')
# define the loss function
# model = GloveSimGeneral(
model = GloveSimGeneral(
	model = transe,
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size(),
	sim_regul_rate = 0.05,
	weight_matrix = sim_weight
)
# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)