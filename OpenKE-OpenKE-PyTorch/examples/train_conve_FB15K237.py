import openke
from openke.config import Trainer, Tester
from openke.module.model import ConvE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

batch_size = 128

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	batch_size = batch_size, 
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
conve = ConvE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
    hidden = 9728, #??
	dim = 200, 
    embedding_shape = 20,
    drops = [0.3, 0.3, 0.2]
    )


# define the loss function
model = NegativeSampling(
	model = conve, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
conve.save_checkpoint('./checkpoint/conve.ckpt')

# test the model
conve.load_checkpoint('./checkpoint/conve.ckpt')
tester = Tester(model = conve, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)