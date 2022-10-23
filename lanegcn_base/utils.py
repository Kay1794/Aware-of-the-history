# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import sys
import cv2
import os
from torch import optim
from matplotlib import pyplot as plt
from copy import deepcopy as cp
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
import time
import random

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def graph_visualize(graphs,ct):
    batch_size = len(graphs)

    
    for i in range(batch_size):
        plt.clf()
        graph_node = graphs[i]["ctrs"].numpy()
        plt.scatter(graph_node[:,0],graph_node[:,1],color=(0.49,0.65,0.88),s=2,zorder=10)
        #for k in range(len(graphs[i]["pre"])):

        for j in range(graphs[i]["pre"][0]["u"].shape[0]):
            u = graphs[i]["pre"][0]["u"][j]
            v = graphs[i]["pre"][0]["v"][j]
            plt.plot([graph_node[u][0],graph_node[v][0]],[graph_node[u][1],graph_node[v][1]],color='k',zorder=5,alpha=0.5)


        plt.savefig('./map_vis/'+str(ct)+'.png')
        ct = ct + 1
        
    return ct

def prob_map_visualize(graphs,prob_maps,actors,futures):
    batch_size = len(graphs)

    
    for i in range(batch_size):
        plt.clf()
        actor = actors[i]
        future = futures[i].cpu().numpy()
        prob_map = torch.max(prob_maps[i],-1)[0].cpu().numpy().astype(np.bool)

        graph_node = graphs[i]["ctrs"].numpy()
        plt.scatter(graph_node[~prob_map][:,0],graph_node[~prob_map][:,1],color='r',s=2,zorder=10)
        plt.scatter(graph_node[prob_map][:,0],graph_node[prob_map][:,1],color='g',s=2,zorder=10)

        for j in range(graphs[i]["pre"][0]["u"].shape[0]):
            u = graphs[i]["pre"][0]["u"][j]
            v = graphs[i]["pre"][0]["v"][j]
            plt.plot([graph_node[u][0],graph_node[v][0]],[graph_node[u][1],graph_node[v][1]],color='k',zorder=5,alpha=0.5)

            #plt.scatter(actor[:,0],actor[:,1],color = 'c',marker="*",s=10,zorder=20)

            for a_id in range(future.shape[0]):
                plt.scatter(future[a_id,-1,0],future[a_id,-1,1],s=10,marker='*',zorder=15,c='y')

        plt.savefig('test.png')
        exit()
        
    return 0

def compare_visualize(graphs,pred,ct,savedict):
    batch_size = len(graphs)
    graph_idc = pred['idc']
    for i in range(batch_size):
        plt.clf()
        plt.subplot(1, 2, 1)
        graph_node = graphs[i]["ctrs"].numpy()
        plt.scatter(graph_node[:,0],graph_node[:,1],color=(0.49,0.65,0.88),s=2,zorder=10)
        #for k in range(len(graphs[i]["pre"])):

        for j in range(graphs[i]["pre"][0]["u"].shape[0]):
            u = graphs[i]["pre"][0]["u"][j]
            v = graphs[i]["pre"][0]["v"][j]
            plt.plot([graph_node[u][0],graph_node[v][0]],[graph_node[u][1],graph_node[v][1]],color='k',zorder=5,alpha=0.5)

        plt.subplot(1, 2, 2)
        graph_node = pred['reg'][graph_idc[i]].cpu().numpy()
        
        plt.scatter(graph_node[:,0],graph_node[:,1],color=(1.00,0.70,0.10),s=2,zorder=10)

        plt.savefig(savedict+'/'+str(ct)+'.png')
        ct = ct + 1
        if ct > 100:
            return ct
    return ct

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path

    
def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 1e-6] = 1e-6 # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def index_dict(data, idcs):
    returns = dict()
    for key in data:
        returns[key] = data[key][idcs]
    return returns


def rotate(xy, theta):
    st, ct = torch.sin(theta), torch.cos(theta)
    rot_mat = xy.new().resize_(len(xy), 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct
    xy = torch.matmul(rot_mat, xy.unsqueeze(2)).view(len(xy), 2)
    return xy


def merge_dict(ds, dt):
    for key in ds:
        dt[key] = ds[key]
    return


class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def update_lr(self,epoch):
        if self.clip_grads:
            self.clip()
        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        BCE_loss = torch.nn.BCELoss()(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss




def result_plot(map_nodes,mod_output,final_output,gt,label,origin,idx):
    color = ['#ECD98B','#AAAAC2','#03875C','#9A4C43','#077ABD','c']
    plt.clf()
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.axis('equal')
        plt.xlim(origin[0]-50,origin[0]+50)
        plt.ylim(origin[1]-50,origin[1] + 50)
        plt.scatter(map_nodes[:,0],map_nodes[:,1],c='k',alpha=.1,s=1,zorder=1)

    for i in range(mod_output.shape[0]):

        if label[i] < 0:
            label[i] = 6
            c = 'b' 
        else:
            c = color[int(label[i])]
        plt.subplot(2,4,int(label[i])+1)
        plt.plot(mod_output[i,:,0],mod_output[i,:,1],zorder=2,c=c,alpha=.3)
    for i in range(final_output.shape[0]):
        c = color[i]
        plt.subplot(2,4,i+1)
        plt.plot(final_output[i,:,0],final_output[i,:,1],zorder=3,c=c)

    plt.subplot(2,4,8)
    for i in range(final_output.shape[0]):
        if i == 0:
            c = 'r'
        else:
            c = 'y'
        plt.plot(final_output[i,:,0],final_output[i,:,1],zorder=3,c=c,alpha=.5)
    plt.plot(gt[0,:,0],gt[0,:,1],zorder=4,c='g',alpha=.5)

    plt.savefig('cluster_vis/'+str(idx)+'.png',dpi=1200)
    return




    
