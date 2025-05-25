# Mimicking To Dominate: Imitation Learning Strategies for Success in Multiagent Competitive Games

## Introduction

Training agents in multi-agent competitive games presents significant challenges due to their intricate nature. These challenges are exacerbated by dynamics influenced not only by the environment but also by opponents' strategies. Existing methods often struggle with slow convergence and instability. To address this, we harness the potential of imitation learning (IL) to comprehend and anticipate opponents' behavior, aiming to mitigate uncertainties with respect to the game dynamics. 

Our key contributions include:
- (i) A new multi-agent imitation learning model for predicting the next moves of opponents. Our model works with hidden opponents' actions and local observations. 
- (ii) A new multi-agent reinforcement learning (MARL) algorithm that combines our imitation learning model and policy training into a single training process. 
- (iii) Extensive experiments in three challenging game environments, including an advanced version of the StarCraft multi-agent challenge (SMACv2), Google Research Football (GRF), and Gold Miner. 

Experimental results show that our approach, IMAX-PPO (Imitation-enhanced Multi-Agent EXtended PPO), achieves superior performance compared to existing state-of-the-art multi-agent RL algorithms. 

## Requirements

- PyTorch version 2.0+: [https://pytorch.org/get-started/](https://pytorch.org/get-started/)
- SMACv2: [https://github.com/oxwhirl/smacv2/](https://github.com/oxwhirl/smacv2/)
- GRF: [https://github.com/google-research/football/](https://github.com/google-research/football/)
- Miner: [https://github.com/xphongvn/rlcomp2020/](https://github.com/xphongvn/rlcomp2020/) or via the folder `./miner/` of this repository.
- Miner's heuristic agents: Go to `./miner/heuristic/` and install by `pip install .`

## Code Structure

```
├── envs
│   ├── base_env                # Multi-agent environments
│   ├── smac_env                # SMACv2 wrapper
│   ├── grf_env                 # GRF wrapper
│   └── miner_env               # Miner wrapper
├── models
│   ├── agents                  # Multi-agent algorithms
│   │   ├── imaxppo.py          # Our IMAX method
│   │   └── mappo.py            # MAPPO method
│   ├── policies
│   │   └── policy.py           # Actor-critic policy
│   └── utils.py
├── trainer
│   ├── buffer.py               # Replay buffer
│   ├── runner.py               # Trajectory collector
│   └── trainer.py              # Multi-agent trainer
├── configs                     # Configuration
├── main.py
└── utils.py
```

## Training Our Methods from Scratch

### SMACv2
```bash
python -u main.py --sc2-path=[SC2_PATH] --map-name [MAP_NAME]
```
- `[SC2_PATH]` is the directory of the downloaded and extracted StarCraftII environment.
- `[MAP_NAME]` is the name of the SMACv2 scenario.

### GRF
```bash
python -u main.py --map-name [GRF_NAME] --n-timesteps=1000000 --ent-coef=0.0
```
- `[GRF_NAME]` is the name of the Google Research Football scenario.
- Supported scenarios: `academy_3_vs_1_with_keeper`, `academy_counterattack_easy`, and `academy_counterattack_hard`.
- Other scenarios might also work.

### Miner
```bash
python -u main.py --map-name [MINER_NAME] --n-envs=32
```
- `[MINER_NAME]` is the name of the Miner scenario.
- More details of this environment are listed in the folder `./miner`.

## Evaluation and Analysis

- To compare our method with existing approaches, please check the notebook `./benchmark.ipynb` for analysis and visualization.
- Check our paper for more details on hyperparameter settings, configuration, and experimental results.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{bui2024mimicking,
  title={Mimicking to dominate: Imitation learning strategies for success in multiagent competitive games},
  author={Bui, The Viet and Mai, Tien and Nguyen, Hong Thanh},
  booktitle={Proceedings of 38th Annual Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024},
  month={December},
  address={Vancouver, Canada}
}
```

## Contact

For questions or inquiries, please contact:
- The Viet Bui: `theviet.bui.2023@phdcs.smu.edu.sg` 
- Tien Mai: `atmai@smu.edu.sg` 
- Thanh Hong Nguyen: `thanhhng@cs.uoregon.edu` 
