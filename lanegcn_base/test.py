# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ArgoTestDataset
from utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn_lba", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)

#new config info
parser.add_argument(
    "--target_weight", default=0., type=float, help="Target weight"
)
parser.add_argument(
    "-c", "--comment", default="lba", type=str,  help="save folder name"
)
parser.add_argument("--n_cluster", default=10, type=int)
parser.add_argument("--L_o_weight", default=10., type=float)
parser.add_argument("--use_time_weight", action="store_true")
parser.add_argument('--lr', nargs="*", type=float,default=[1e-3,1e-4], help='MultistepLR parameter')
parser.add_argument('--step', nargs="*", type=int,default=[32], help='MultistepLR parameter')
parser.add_argument("--use_mask", action="store_true")
parser.add_argument("--soft_adj", action="store_true")
parser.add_argument("--learn_adj", action="store_true")
parser.add_argument("--subset_train", action="store_true")
parser.add_argument("--weighted_loss", action="store_true")
parser.add_argument("--displacement", action="store_true")
parser.add_argument(
    "--behavior_root", default="ActorSemanticGraph", type=str,  help="model type")
parser.add_argument("--use_brier", action="store_true")
parser.add_argument("--nepoch", default=0,type=int)
parser.add_argument("--nbatch", default=32,type=int)
parser.add_argument("--sem_pos_only", action="store_true")
parser.add_argument("--n_behavior", default=40,type=int)
parser.add_argument("--m2m_dist", default=4.,type=float)
parser.add_argument("--s2a_dist", default=.5,type=float)
parser.add_argument("--num_mods", default=6,type=int)
parser.add_argument("--num_out", default=6,type=int)

def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model(args)


    # load pretrain model
    
    #if not os.path.isabs(ckpt_path):
    #    ckpt_path = os.path.join(config["save_dir"], ckpt_path)

    ckpt_path = args.resume
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    prob = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        with torch.no_grad():
            output,output_actornet = net(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
            results_prob = [x[0:1].detach().cpu().numpy() for x in output["cls"]]
        for i, (argo_idx, pred_traj,pred_prob) in enumerate(zip(data["argo_id"], results,results_prob)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None
            prob[argo_idx] = pred_prob.squeeze()

    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    
    # evaluate or submit
    if args.split == "val":
        # for val set: compute metric
        from argoverse.evaluation.eval_forecasting import (
            compute_forecasting_metrics,
        )
        # Max #guesses (K): 6
        _ = compute_forecasting_metrics(preds, gts, cities, 6, 30, 2,prob)
        # Max #guesses (K): 1
        _ = compute_forecasting_metrics(preds, gts, cities, 1, 30, 2,prob)
    else:
        # for test set: save as h5 for submission in evaluation server
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5",probabilities=prob)  # this might take awhile
    #import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    main()
