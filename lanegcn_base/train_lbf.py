# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
import horovod.torch as hvd


from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain

from mpi4py import MPI
import gc


comm = MPI.COMM_WORLD
hvd.init()
torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
	"-m", "--model", default="lanegcn_lbf", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
	"--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
	"--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument(
	"--teacher_dir", default="", type=str, metavar="RESUME", help="checkpoint path"
)
#new config info
parser.add_argument(
	"--target_weight", default=0., type=float, help="Target weight"
)
parser.add_argument(
	"-c", "--comment", default="lbf", type=str, help="save folder name"
)

parser.add_argument("--n_cluster", default=36, type=int)
parser.add_argument("--L_o_weight", default=10., type=float)
parser.add_argument("--use_double_actornet", action="store_true")
parser.add_argument("--double_actornet_weight", default=0.5, type=float)
parser.add_argument("--use_time_weight", action="store_true")
parser.add_argument('--lr', nargs="*", type=float,default=[1e-3,1e-4], help='MultistepLR parameter')
parser.add_argument('--step', nargs="*", type=int,default=[32], help='MultistepLR parameter')
parser.add_argument("--use_mask", action="store_true")
parser.add_argument("--rot_aug", action="store_true")
parser.add_argument("--spectral_weight", default=1.,type=float)
parser.add_argument("--load_opt", default=1,type=int)

parser.add_argument("--subset_train", action="store_true")
parser.add_argument("--weighted_loss", action="store_true")
parser.add_argument("--displacement", action="store_true")
parser.add_argument("--use_std", action="store_true")
parser.add_argument("--use_brier", action="store_true")
parser.add_argument("--nepoch", default=36,type=int)
parser.add_argument("--nbatch", default=32,type=int)
parser.add_argument("--m2m_dist", default=4.,type=float)
parser.add_argument("--n_behavior", default=40,type=int)
parser.add_argument("--sem_pos_only", action="store_true")
parser.add_argument("--zero_set", action="store_true")
parser.add_argument("--all_zeros", action="store_true")
parser.add_argument("--add_rot_info", action="store_true")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument(
	"--behavior_method", default="m2a", type=str,  help="How to add Behavior info"
)
parser.add_argument(
	"--behavior_root", default="", type=str,  help="dir to behavior database")
parser.add_argument(
	"--behavior_loss", default="l2", type=str,  help="loss type for KD learning")

parser.add_argument(
	"--save_dir", default="./results", type=str,  help="How to add Behavior info"
)
parser.add_argument("--kd_weight", default=1.,type=float)
parser.add_argument("--learn_after_a2a", action="store_true")
parser.add_argument("--kl_reduce", default="mean", type=str)
parser.add_argument("--post_encoder", action="store_true")
parser.add_argument("--kd_decay", action="store_true")
parser.add_argument("--implicit", action="store_true")
parser.add_argument("--wo_relu", action="store_true")
parser.add_argument("--double_kd", action="store_true")
parser.add_argument("--dest_kd", action="store_true")
parser.add_argument("--s2a_dist", default=.5,type=float)
parser.add_argument("--behavior_min_size", default=0.,type=float)
def main():
	seed = hvd.rank()
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	# Import all settings for experiment.
	args = parser.parse_args()
	model = import_module(args.model)
	config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(args)
	if hvd.rank() == 0:
	  print(sys.argv[0:])
	  print(args.__repr__() + "\n\n")

	if config["horovod"]:
		if args.pretrain:
			opt.opt = hvd.DistributedOptimizer(
				opt.opt, named_parameters=net.named_parameters()
			)
			for parameter in net.teacher.parameters():
				parameter.requires_grad = False
			pretrain = True
		else:
			opt.opt = hvd.DistributedOptimizer(
				opt.opt, named_parameters=net.named_parameters()
			)
			pretrain = False

	if args.resume or args.weight:
		ckpt_path = args.resume or args.weight
		#ckpt_path = os.path.join(ckpt_path)
		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		load_pretrain(net, ckpt["state_dict"])
		if args.resume:
			config["epoch"] = ckpt["epoch"]
			opt.load_state_dict(ckpt["opt_state"])
	if pretrain:
		print("Lodaing Pretrained Teacher...")

		ckpt_path = args.teacher_dir
		print('Load teacher net from ',ckpt_path)
		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		load_pretrain(net.teacher,ckpt["state_dict"])

	if args.eval:
		# Data loader for evaluation
		dataset = Dataset(config["val_split"], config, train=False)
		val_sampler = DistributedSampler(
			dataset, num_replicas=hvd.size(), rank=hvd.rank()
		)
		val_loader = DataLoader(
			dataset,
			batch_size=config["val_batch_size"],
			num_workers=config["val_workers"],
			sampler=val_sampler,
			collate_fn=collate_fn,
			pin_memory=True,
		)

		hvd.broadcast_parameters(net.state_dict(), root_rank=0)
		val(config, val_loader, net, loss, post_process, 999)
		return

	# Create log and copy all code
	if hvd.rank() == 0:
		save_dir = config["save_dir"]
		log = os.path.join(save_dir, "log")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		sys.stdout = Logger(log)
	if args.resume == '' and hvd.rank()==0:
		
		src_dirs = [root_path]
		dst_dirs = [os.path.join(save_dir, "files")]
		for src_dir, dst_dir in zip(src_dirs, dst_dirs):
			files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
			if not os.path.exists(dst_dir):
				os.makedirs(dst_dir)
			for f in files:
				shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

	# Data loader for training
	dataset = Dataset(config["train_split"], config, train=True)
	#dataset = Dataset(config["val_split"], config, train=True)
	train_sampler = DistributedSampler(
		dataset, num_replicas=hvd.size(), rank=hvd.rank()
	)
	train_loader = DataLoader(
		dataset,
		batch_size=config["batch_size"],
		num_workers=config["workers"],
		sampler=train_sampler,
		collate_fn=collate_fn,
		pin_memory=True,
		worker_init_fn=worker_init_fn,
		drop_last=True,
	)

	# Data loader for evaluation
	dataset = Dataset(config["val_split"], config, train=False)
	val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
	val_loader = DataLoader(
	   dataset,
	   batch_size=config["val_batch_size"],
	   num_workers=config["val_workers"],
	   sampler=val_sampler,
	   collate_fn=collate_fn,
	   pin_memory=True,
	)

	hvd.broadcast_parameters(net.state_dict(), root_rank=0)
	hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

	epoch = config["epoch"]
	remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
	for i in range(remaining_epochs):
		train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader,pretrain)

	val(config, val_loader, net, loss, post_process,config["num_epochs"])

def worker_init_fn(pid):
	np_seed = hvd.rank() * 1024 + int(pid)
	np.random.seed(np_seed)
	random_seed = np.random.randint(2 ** 32 - 1)
	random.seed(random_seed)

#@profile
def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None,pretrain=False):
	train_loader.sampler.set_epoch(int(epoch))
	net.train()

	num_batches = len(train_loader)
	epoch_per_batch = 1.0 / num_batches
	save_iters = int(np.ceil(config["save_freq"] * num_batches))
	display_iters = int(
		config["display_iters"] / (hvd.size() * config["batch_size"])
	)
	val_iters = int(config["val_iters"] / (hvd.size() * config["batch_size"]))

	start_time = time.time()
	student_metrics = dict()
	teacher_metrics = dict()
	for i, data in tqdm(enumerate(train_loader),disable=hvd.rank()):
		epoch += epoch_per_batch
		data = dict(data)

		teacher_out,student_out,behavior_target,behavior_student = net(data)
		#istudent_out['neighbor_num'] = teacher_out['neighbor_num']
		teacher_loss_out = loss(teacher_out,data,student=False)
		#extra info from teacher loss computation
		student_out['teacher_row_idcs'] = teacher_loss_out['teacher_row_idcs']
		student_out['teacher_min_idcs'] = teacher_loss_out['teacher_min_idcs']
		student_loss_out = loss(student_out, data,student=True,behavior_pred=behavior_student,behavior_gt=behavior_target,epoch=epoch)
		

		student_post_out = post_process(student_out, data,student=True)
		teacher_post_out = post_process(teacher_out, data,student=False)

		post_process.append(student_metrics, student_loss_out, student_post_out,student=True)
		post_process.append(teacher_metrics, teacher_loss_out, teacher_post_out,student=False)

		opt.zero_grad()
		student_loss_out["student_loss"].backward(retain_graph=True)
		if not pretrain:
			teacher_loss_out["teacher_loss"].backward()
		lr = opt.step(epoch)

		num_iters = int(np.round(epoch * num_batches))
		if hvd.rank() == 0 and (
			num_iters % save_iters == 0 or epoch >= config["num_epochs"]
		):
			save_ckpt(net, opt, config["save_dir"], epoch)

		if num_iters % display_iters == 0:
			dt = time.time() - start_time
			student_metrics = sync(student_metrics)
			teacher_metrics = sync(teacher_metrics)
			if hvd.rank() == 0:
				post_process.display(student_metrics, dt, epoch, lr,student=True)
				post_process.display(teacher_metrics, dt, epoch, lr,student=False)
			start_time = time.time()
			student_metrics = dict()
			teacher_metrics = dict()

		if num_iters % val_iters == 0:
		   val(config, val_loader, net, loss, post_process, epoch)




def val(config, data_loader, net, loss, post_process, epoch,):
	net.eval()

	start_time = time.time()
	student_metrics = dict()
	teacher_metrics = dict()
	for i, data in tqdm(enumerate(data_loader)):
		data = dict(data)
		with torch.no_grad():
			
			teacher_out,student_out,behavior_target,behavior_student = net(data)
			student_loss_out = loss(student_out, data,student=True,behavior_pred=behavior_student,behavior_gt=behavior_target)
			
			teacher_loss_out = loss(teacher_out,data,student=False)
			student_post_out = post_process(student_out, data,student=True)
			teacher_post_out = post_process(teacher_out, data,student=False)

			post_process.append(student_metrics, student_loss_out, student_post_out,student=True)
			post_process.append(teacher_metrics, teacher_loss_out, teacher_post_out,student=False)

	dt = time.time() - start_time
	metrics = sync(metrics)
	if hvd.rank() == 0:
		post_process.display(student_metrics, dt, epoch,student=True)
		post_process.display(teacher_metrics, dt, epoch,student=False)
	net.train()


def save_ckpt(net, opt, save_dir, epoch):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	state_dict = net.state_dict()
	for key in state_dict.keys():
		state_dict[key] = state_dict[key].cpu()

	save_name = "%3.3f.ckpt" % epoch
	torch.save(
		{"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
		os.path.join(save_dir, save_name),
	)


def sync(data):
	data_list = comm.allgather(data)
	data = dict()
	for key in data_list[0]:
		if isinstance(data_list[0][key], list):
			data[key] = []
		else:
			data[key] = 0
		for i in range(len(data_list)):
			data[key] += data_list[i][key]
	return data


if __name__ == "__main__":
	main()
