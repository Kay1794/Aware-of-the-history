## Requirement

### Recommend Environment

* Tested OS: Linux / RTX 3090
* Python == 3.7.11
* PyTorch == 1.8.1+cu111

## Data preparation & Environment:

This code is based on the official code of LaneGCN ([Paper](https://arxiv.org/pdf/2007.13732.pdf); [Github](https://github.com/uber-research/LaneGCN)). PLease follow the instructions from LaneGCN official github to install the relative packages and the preprocessed Argoverse Dataset[link](https://www.argoverse.org/av1.html#download-link). Then please download and unzip our preprocessed Argoverse Behavior database from this link [link](https://drive.google.com/file/d/1GmnCd4ByA9s6b9RKgQHD3wipQn-MP91u/view?usp=sharing).

## Command:

LBA training script:  
```bash
    $ HOROVOD_CACHE_CAPACITY=2048 CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 -H localhost:4 python train.py --nepoch 36 --comment lba --model lanegcn_lba --behavior_root PATH_TO_BEHAVIOR_DATABASE
```

LBA testing script: 
```bash
    $ CUDA_VISIBLE_DEVICES=0 python test.py --nepoch 36 --comment lba --model lanegcn_lba --resume results/lba/36.000.ckpt --split test --eval --behavior_root PATH_TO_BEHAVIOR_DATABASE
```

LBF training script: 
```bash
    $ HOROVOD_CACHE_CAPACITY=2048 CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 -H localhost:4 python train_lbf.py --nepoch 36 --pretrain --teacher_dir ./results/lba/36.000ckpt --comment lba --model lanegcn_lbf --double_kd --dest_kd --kd_weight 1.5 --behavior_root PATH_TO_BEHAVIOR_DATABASE
```

LBF testing script: 
```bash
    $ CUDA_VISIBLE_DEVICES=4 python test_kd.py --nepoch 36 --comment lbf  --resume ./results/lbf/36.000.ckpt --behavior_root PATH_TO_BEHAVIOR_DATABASE --split test --eval
```
