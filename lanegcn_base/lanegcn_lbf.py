 # Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null,MLP
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from matplotlib import pyplot as plt
import gc
from copy import deepcopy

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()

####
config['behavior'] = True
config['grid'] = False
# config['grid_range'] = (-49.5,49.5,-49.5,49.5)
# config['grid_size'] = (120,120)

config['grid_range'] = (-49.5,99.5,-44.5,44.5)
config['grid_size'] = (150,90)
"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 5
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 44
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 32
config["val_batch_size"] = 32
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join('/GPFS/data/sihengchen/model_result/argo_lanegcn/', "dataset/test_obs/data")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path ,"val_crs_dist6_angle90.p"
)
#config['preprocess_test'] = os.path.join(root_path, 'test_test.p')
config['preprocess_test'] = os.path.join('/GPFS/data/sihengchen/argo/lanegcn/', 'test_test.p')
"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["n_behavior"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["map2map_dist"] = 4.0
config["behavior2actor_dist"] = .5
config["actor2actor_dist"] = 100.0
config["pred_size"] = 30
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2

config["z_dim"] = 16
config["f_dim"] = 16
config['dec_size'] = [1024,512,1024]
config['enc_dest_size'] = [8,16]
config['enc_latent_size'] = [8,50]
config['enc_past_size'] = [512,256]
config['sigma'] = 1.3
config['behavior_min_size'] = 0
config['use_polar'] = False
config['use_actor_only'] = False
config['use_map_only'] = False
### end of config ###

### new config info ###
config['double_actornet']=False
config['target_weight'] = 0.

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.teacher = TeacherNet(config)
        self.student = StudentNet(config)

    def forward(self, data: Dict,training=True) -> Dict[str, List[Tensor]]:
    
    
        if training:
            teacher_out,behavior_target = self.teacher(deepcopy(data))
            student_out,behavior_student = self.student(deepcopy(data))
            student_out['neighbor_num'] = teacher_out['neighbor_num']

            return teacher_out,student_out,behavior_target,behavior_student
        else:
            student_out,behavior_student = self.student(deepcopy(data))

            return student_out


class TeacherNet(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(TeacherNet, self).__init__()
        self.config = config


        self.actor_net = ActorNet(config)
        self.map_net = MapNetBehavior(config)

        self.a2m = A2M(config)
        self.behavior = BehaviorNet(config)
        self.pre_a2a = A2A(config)
        self.behavior_fuse = BehaviorM2A(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.pred_net = PredNet(config)

    def forward(self, data: Dict,training=True) -> Dict[str, List[Tensor]]:
        #if self.config['double_kd']:
        behavior_target = []
        # construct actor feature

        actors, actor_idcs = actor_gather(gpu(data["feats"]))

        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        out_actor = None

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))

        # construct behavior features
        behavior = behavior_gather(to_long(gpu(data["behavior"])))
        
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        # actor-map fusion cycle 
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)

        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        
        
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        behavior_nodes,behavior_ctrs,behavior_idcs = self.behavior(behavior)
        actors,neighbor_num = self.behavior_fuse(actors, actor_idcs, actor_ctrs, behavior_nodes, behavior_idcs, behavior_ctrs)
        
        if self.config['double_kd']:
            behavior_target.append(actors.clone())
        actors = self.pre_a2a(actors, actor_idcs, actor_ctrs,feature_extract=self.config['double_kd'])
        if self.config['double_kd']:
            behavior_target.append(actors.clone())

        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        out['neighbor_num'] = neighbor_num.clone()
        if self.config['dest_kd'] and self.config['double_kd']:
            behavior_target.append(out['att_feat'].clone())

        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        return out,behavior_target

class StudentNet(nn.Module):
    """
    Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction header for motion forecasting using 
           feature from A2A
    """
    def __init__(self, config):
        super(StudentNet, self).__init__()
        self.config = config


        self.actor_net = ActorNet(config)
        self.map_net = MapNetBehavior(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)

        self.transfer_net = TransferNet(config)
        self.behavior_fuse = BehaviorM2A(config)
        self.pre_a2a = A2A(config)

        self.pred_net = PredNet(config)
    def forward(self, data: Dict,training=True) -> Dict[str, List[Tensor]]:
        if self.config['double_kd']:
            behavior_student = []
        # construct actor feature

        actors, actor_idcs = actor_gather(gpu(data["feats"]))

        actor_ctrs = gpu(data["ctrs"])
        actors = self.actor_net(actors)
        if self.config['use_actor_only']:
            actors_ori = actors.clone()


        out_actor = None

        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))

        
        nodes, node_idcs, node_ctrs = self.map_net(graph)
        if self.config['use_map_only']:
            nodes_ori = nodes.clone()

        # actor-map fusion cycle 
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        behavior_source = nodes.clone()

        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs,feature_extract=False)
        
        #
        behavior_nodes = torch.zeros_like(actors).to(actors.device)

        behavior_nodes = self.transfer_net(behavior_nodes, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        
        actors,_ = self.behavior_fuse(actors, actor_idcs, actor_ctrs, behavior_nodes,actor_idcs, actor_ctrs)
        if self.config['double_kd']:
            behavior_student.append(actors.clone())
        elif not self.config['learn_after_a2a']:
            behavior_student = actors.clone()
        actors = self.pre_a2a(actors, actor_idcs, actor_ctrs,feature_extract= self.config['double_kd'])
        if self.config['double_kd']:
            behavior_student.append(actors.clone())
        elif self.config['learn_after_a2a']:
            behavior_student = actors.clone()


        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        if self.config['dest_kd'] and self.config['double_kd']:
            behavior_student.append(out['att_feat'].clone())
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )

        return out,behavior_student


def behavior_gather(behaviors):
    batch_size = len(behaviors) #one scene one graph
    #print(batch_size)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + behaviors[i]["feats"].shape[0]).to(
            behaviors[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + behaviors[i]["feats"].shape[0]

        
    behavior = dict()
    behavior["idcs"] = node_idcs
    behavior["ctrs"] = [x["ctrs"] for x in behaviors]
    behavior["feats"] = torch.cat([x['feats'] for x in behaviors], 0)


    return behavior


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    batch_size = len(actors)
    num_actors = [len(x) for x in actors] #x: [20:]


    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs

def dest_gather(gts: List[Tensor], rot: Tensor, orig: Tensor,has_preds:List[Tensor] ) -> Tuple[Tensor, List[Tensor]]:

    # transform prediction to world coordinates
    for i in range(len(gts)):
        rot_matrix = gpu(rot[i].unsqueeze(0).repeat(gts[i].shape[0],1,1))
        gts[i] = torch.bmm(rot_matrix,(gts[i]-orig[i].view(1, 1, -1)).transpose(1,2)).transpose(1,2)
        gts[i][~has_preds[i]] = 0.
        
    batch_size = len(gts)
    num_actors = [len(x) for x in gts] #x: [30,2]


    dest = [x.transpose(1, 2)[:,:,-1] for x in gts]
    #dest = [x.transpose(1, 2) for x in gts]
    dest = torch.cat(dest, 0)


    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(dest.device)
        actor_idcs.append(idcs)
        count += num_actors[i]

    return dest, actor_idcs


def graph_gather(graphs):
    batch_size = len(graphs) #one scene one graph
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]
    #graph["behavior"] = torch.cat([x["behavior"] for x in graphs],dim=0).type(torch.float32)

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)
        



    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])): # 6
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )


    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph

class BehaviorActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(BehaviorActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1
        n_in = 2
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, behavior: Tensor) -> Tensor:
        behavior_ctrs = behavior["ctrs"]
        behavior_idcs = behavior["idcs"]
        behavior_feat = behavior['feats']
        out = behavior_feat.permute(0,2,1) #[N,2,20]
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out,behavior_ctrs,behavior_idcs

class BehaviorNet(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(BehaviorNet, self).__init__()
        self.config = config
        n_feat = config["n_map"]
        n_sem = config['n_behavior']
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(n_sem, n_feat),
            nn.ReLU(inplace=True),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),

        )


    def forward(self, behavior):

        behavior_ctrs = behavior["ctrs"]
        behavior_idcs = behavior["idcs"]
        feat = self.input(behavior['feats'].reshape(-1,config['n_behavior']))
        
        return feat,behavior_ctrs,behavior_idcs





class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1


        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)



    def forward(self, actors: Tensor) -> Tensor:
        out = actors #[N,3,20]
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class MapNetBehavior(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(MapNetBehavior, self).__init__()
        self.config = config
        self.use_project = False
        norm = "GN"
        ng = 1

        n_map = config["n_map"]
        final_map = n_map
        n_behavior = config["n_behavior"]


        self.in_dim = 2
        self.input = nn.Sequential(
            nn.Linear(self.in_dim, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(self.in_dim, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )


        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, final_map), final_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(final_map, final_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(final_map, final_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)

        feat = self.input(ctrs)
            
        feat += self.seg(graph["feats"])
        
        feat = self.relu(feat)
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])): 
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]



class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """
    def __init__(self, config,dest=False):
        super(A2M, self).__init__()
        self.config = config
        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor],use_relu: bool=True) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
                use_relu
            )
        return feat




class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """
    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"] 
        norm = "GN"
        ng = 1



        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict,use_relu: bool=True,behavior=None) -> Tensor:
        """fuse map"""

        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            if use_relu:
                    feat = self.relu(feat)
            res = feat
        return feat

class TransferNet(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config):
        super(TransferNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1


        n_actor = config["n_actor"]

        n_map = config["n_map"]
        n_feat = config["n_map"]
        self.encode = nn.Sequential(
            nn.Linear(n_map, n_feat),
            nn.ReLU(inplace=True),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),
            LinearRes(n_feat, n_feat, norm=norm, ng=ng),

        )
        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, behavior: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor],use_relu: bool=True) -> Tensor:
        
        nodes = self.encode(nodes)
        for i in range(len(self.att)):
            
            behavior_nodes = self.att[i](
                behavior,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
                use_relu
            )
        return behavior_nodes


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config,dest = False):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]

        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor],use_relu: bool=True) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
                use_relu
            )
        return actors

class BehaviorM2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config,dest = False):
        super(BehaviorM2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]

        n_sem = config["n_map"] # after BehaviorNet, n_sem == n_map
        att = []
        for i in range(2):
            att.append(Att(n_actor, n_sem))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor],use_relu: bool=True) -> Tensor:
        
        for i in range(len(self.att)):

            relu = True
            actors,neighbor_num = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["behavior2actor_dist"],
                use_relu=relu,
                get_mask=True
            )
        return actors,neighbor_num



class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config,dest=False):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor],use_relu: bool=True,
        feature_extract=False) -> Tensor:
        for i in range(len(self.att)):
            if feature_extract and (i == len(self.att)-1):
                relu = False
            else:
                relu = True
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
                use_relu=relu
            )
        return actors


class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self, config,dest=False):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor],dest: Tensor=None,gt_dest: Tensor=None) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        att_feat = feats.clone().view(-1,self.config['num_mods'],self.config['n_actor'])
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        out["dest"],out["gt_dest"] = [], []

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
            if not dest is None:
                out["dest"].append(dest[idcs])
                out["gt_dest"].append(gt_dest[idcs])
        out['att_feat'] = att_feat

        return out


class KDPredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """
    def __init__(self, config,dest=False):
        super(KDPredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        if dest:
            n_actor = config["n_actor"] + config["f_dim"] + 3
        else:
            n_actor = config["n_actor"]

        pred_encoder = []
        pred_header = []
        for i in range(config["num_mods"]):
            pred_encoder.append(LinearRes(n_actor, n_actor, norm=norm, ng=ng))
            pred_header.append(nn.Linear(n_actor, 2 * config["num_preds"]))
            
        self.pred_encoder = nn.ModuleList(pred_encoder)
        self.pred_header = nn.ModuleList(pred_header)
        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor],dest: Tensor=None,gt_dest: Tensor=None) -> Dict[str, List[Tensor]]:
        preds = []
        feature_kd_list = []
        for i in range(len(self.pred)):
            feature = self.pred_encoder[i](actors)
            feature_kd_list.append(feature.clone())
            preds.append(self.pred_header(feature))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        out["dest"],out["gt_dest"] = [], []
        out["kd_feature"] = []
        feature_kd_list= torch.stack(feature_kd_list,axis=1)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
            if not dest is None:
                out["dest"].append(dest[idcs])
                out["gt_dest"].append(gt_dest[idcs])
                out["kd_feature"].append(feature_kd_list[idcs])
        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float,use_relu: bool=True,get_mask:bool=False) -> Tensor:
        res = agts
        if len(ctx) == 0:
            
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            if get_mask:
                neighbor_num = torch.zeros(agts.shape[0])
                return agts,neighbor_num
            else:
                return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        if get_mask:
            neighbor_num = []
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2)+ 1e-6)
            mask = dist <= dist_th
            
            idcs = torch.nonzero(mask, as_tuple=False)
            if get_mask:
                neighbor_num.append(torch.sum(mask,-1))
            if len(idcs) == 0:
                continue
            

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        if use_relu:
            agts = self.relu(agts)

        if get_mask:
            neighbor_num = torch.cat(neighbor_num)
            return agts,neighbor_num
        else:
            return agts


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts




class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], gt_preds: List[Tensor], has_preds: List[Tensor],student=False) -> Dict[str, Union[Tensor, int]]:
        if student:
            prefix ='student_'
        else:
            prefix = 'teacher_'
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out[prefix+"cls_loss"] = zero.clone()
        loss_out[prefix+"num_cls"] = 0
        loss_out[prefix+"reg_loss"] = zero.clone()
        loss_out[prefix+"num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out[prefix+"cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out[prefix+"num_cls"] += mask.sum().item()

        
        reg = reg[row_idcs, min_idcs]
        loss_out[prefix+'row_idcs'] = row_idcs.clone()
        loss_out[prefix+'min_idcs'] = min_idcs.clone()
        coef = self.config["reg_coef"]
        loss_out[prefix+"reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out[prefix+"num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)
        if config['behavior_loss'] == 'l2':
            self.behavior_reg_loss = nn.MSELoss(reduction="mean")
        elif config['behavior_loss'] == 'kl':
            self.behavior_reg_loss = nn.KLDivLoss(reduction=config['kl_reduce'] )

    def forward(self, out: Dict, data: Dict,student=False,behavior_pred=None,behavior_gt=None,epoch=-1) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]),student=student)
        if student:
            prefix ='student_'
            neighbor_num_mask = out['neighbor_num'] >= self.config['behavior_min_size']
            if self.config['behavior_loss'] == 'kl':
                if self.config['double_kd']:
                    loss_out['behavior_recon'] = self.behavior_reg_loss(F.log_softmax(behavior_pred[0][neighbor_num_mask],dim=-1),F.softmax(behavior_gt[0][neighbor_num_mask],dim=-1))\
                    + self.behavior_reg_loss(F.log_softmax(behavior_pred[1][neighbor_num_mask],dim=-1),F.softmax(behavior_gt[1][neighbor_num_mask],dim=-1))
                    if self.config['dest_kd']:
                        teacher_row_mask = neighbor_num_mask[out['teacher_row_idcs']] > 0
                        student_row_mask = neighbor_num_mask[loss_out['student_row_idcs']] > 0

                        behavior_pred[2] = behavior_pred[2][loss_out['student_row_idcs'][student_row_mask],loss_out['student_min_idcs'][student_row_mask]]
                        behavior_gt[2] = behavior_gt[2][out['teacher_row_idcs'],out['teacher_min_idcs']]
                        loss_out['behavior_recon'] += self.behavior_reg_loss(F.log_softmax(behavior_pred[2],dim=-1),F.softmax(behavior_gt[2],dim=-1))
                else:
                    loss_out['behavior_recon'] = self.behavior_reg_loss(F.log_softmax(behavior_pred[neighbor_num_mask],dim=-1),F.softmax(behavior_gt[neighbor_num_mask],dim=-1))
            else:
                if self.config['double_kd']:
                    loss_out['behavior_recon'] = self.behavior_reg_loss(behavior_pred[0][neighbor_num_mask],behavior_gt[0][neighbor_num_mask]) \
                    + self.behavior_reg_loss(behavior_pred[1][neighbor_num_mask],behavior_gt[1][neighbor_num_mask])
                    
                    if self.config['dest_kd']:
                        teacher_row_mask = neighbor_num_mask[out['teacher_row_idcs']] > 0
                        student_row_mask = neighbor_num_mask[loss_out['student_row_idcs']] > 0

                        behavior_pred[2] = behavior_pred[2][loss_out['student_row_idcs'][student_row_mask],loss_out['student_min_idcs'][student_row_mask]]
                        behavior_gt[2] = behavior_gt[2][out['teacher_row_idcs'][teacher_row_mask],out['teacher_min_idcs'][teacher_row_mask]]
                        loss_out['behavior_recon'] += self.behavior_reg_loss(behavior_pred[2],behavior_gt[2])
                
                else:
                    loss_out['behavior_recon'] = self.behavior_reg_loss(behavior_pred[neighbor_num_mask],behavior_gt[neighbor_num_mask])
            
        else:
            prefix = 'teacher_'
        loss_out[prefix+"loss"] = loss_out[prefix+"cls_loss"] / (
            loss_out[prefix+"num_cls"] + 1e-10
        ) + loss_out[prefix+"reg_loss"] / (loss_out[prefix+"num_reg"] + 1e-10)
        if student:
            if (self.config['kd_decay'] and (epoch <= self.config["lr_epochs"][0])) or (not self.config['kd_decay']):
                loss_out[prefix+"loss"] += self.config['kd_weight']  * loss_out['behavior_recon']
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data,student=False):
        if student:
            prefix ='student_'
        else:
            prefix = 'teacher_'
        post_out = dict()
        post_out[prefix+"preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out[prefix+"gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out[prefix+"has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None,student=False) -> Dict:
        if student:
            prefix ='student_'
        else:
            prefix = 'teacher_'
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != prefix+"loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == prefix+"loss" or key.find('idcs') > 0:
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None,student=False):
        """Every display-iters print training/val information"""
        if student:
            prefix ='student_'
        else:
            prefix = 'teacher_'
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics[prefix+"cls_loss"] / (metrics[prefix+"num_cls"] + 1e-10)
        reg = metrics[prefix+"reg_loss"] / (metrics[prefix+"num_reg"] + 1e-10)
        if student:
            recon = metrics['behavior_recon']
        else:
            recon = 0.
        loss = cls + reg

        preds = np.concatenate(metrics[prefix+"preds"], 0)
        gt_preds = np.concatenate(metrics[prefix+"gt_preds"], 0)
        has_preds = np.concatenate(metrics[prefix+"has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            prefix[:-1] + ": loss %2.4f %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, recon, ade1, fde1, ade, fde)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model(args):
    #mod config if necessary
    config['behavior_root'] = args.behavior_root
    config["lr"] = args.lr
    config["lr_epochs"] = args.step
    config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])
    print('Load behavior from: ',config['behavior_root'])

    if args.resume != '':
        config["save_dir"] = args.resume[:args.resume.rfind('/')]
    elif args.save_dir != "":
        config["save_dir"] = os.path.join(args.save_dir,args.comment)
    else:
        config["save_dir"] = os.path.join(
            root_path, "results", args.comment
        )
    config['num_epochs'] = args.nepoch
    config['behavior_loss'] = args.behavior_loss
    config["batch_size"] = args.nbatch
    config["val_batch_size"] = config["batch_size"]
    config['n_behavior'] = args.n_behavior
    config['kd_weight'] = args.kd_weight
    config['double_kd'] = args.double_kd
    config['dest_kd'] = args.dest_kd
    config['behavior_min_size'] = args.behavior_min_size

    net = Net(config)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)


    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
