## Aware of the History: Trajectory Forecasting with the Local Behavior Data

**Official PyTorch code** for our ECCV'22 paper [Collaborative Uncertainty in Multi-Agent Trajectory Forecasting](https://arxiv.org/abs/2207.09646).

**Abstract**: The historical trajectories previously passing through a location may help infer the future trajectory of an agent currently at this location. Despite great improvements in trajectory forecasting with the guidance of high-definition maps, only a few works have explored such local historical information. In this work, we re-introduce this information as a new type of input data for trajectory forecasting systems: the local behavior data, which we conceptualize as a collection of location-specific historical trajectories. Local behavior data helps the systems emphasize the prediction locality and better understand the impact of static map objects on moving agents. We propose a novel local-behavior-aware (LBA) prediction framework that improves forecasting accuracy by fusing information from observed trajectories, HD maps, and local behavior data. Also, where such historical data is insufficient or unavailable, we employ a local-behavior-free (LBF) prediction framework, which adopts a knowledge-distillation-based architecture to infer the impact of missing data. Extensive experiments demonstrate that upgrading existing methods with these two frameworks significantly improves their performances. Especially, the LBA framework boosts the SOTA methods' performance on the nuScenes dataset by at least 14% for the K=1 metrics.


## Overview

LBA / LBF implementation for LaneGCN (released) [link]

LBA / LBF implementation for P2T (SOON)


## Acknowledgement

We thanks for the Argoverse data and the code provided by LaneGCN ([Paper](https://arxiv.org/pdf/2007.13732.pdf); [Github](https://github.com/uber-research/LaneGCN)).

## Citation

If you use this code, please cite our papers:

```
@article{zhong2022aware,
  title={Aware of the History: Trajectory Forecasting with the Local Behavior Data},
  author={Zhong, Yiqi and Ni, Zhenyang and Chen, Siheng and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2207.09646},
  year={2022}
}
```