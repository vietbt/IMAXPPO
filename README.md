# Mimicking To Dominate: Imitation Learning Strategies for Success in Multiagent Competitive Games

### Introduction

Training agents in multi-agent competitive games presents significant challenges due to their intricate nature. These challenges are exacerbated by dynamics influenced not only by the environment but also by opponents' strategies. Existing methods often struggle with slow convergence and instability. To address this, we harness the potential of imitation learning to comprehend and anticipate opponents' behavior, aiming to mitigate uncertainties with respect to the game dynamics. Our key contributions include: 

- (i) a new multi-agent imitation learning model for predicting next moves of the opponents: our model works with hidden opponents' actions and local observations; 
- (ii) a new multi-agent reinforcement learning algorithm that combines our imitation learning model and policy training into one single training process; and 
- (iii) extensive experiments in three challenging game environments, including an advanced version of the Star-Craft multi-agent challenge (i.e., SMACv2). 

Experimental results show that our approach achieves superior performance compared to existing state-of-the-art multi-agent RL algorithms.


### Requirements

- PyTorch version 2.0+: https://pytorch.org/get-started/
- SMACv2: https://github.com/oxwhirl/smacv2/
- GRF: https://github.com/google-research/football/
- Miner: https://github.com/xphongvn/rlcomp2020/ or via the folder [./miner/](miner) of this repository
- Miner's heuristic agents: goto [./miner/heuristic/](./miner/heuristic/) and install by `pip install .`

### Code structure

    ├── envs                        
    │   ├── base_env                # Multi-agent environments
    │   ├── smac_env                # SMACv2 wrapper
    │   ├── grf_env                 # GRF wrapper
    │   └── miner_env               # Miner wrapper
    ├── models
    │   ├── agents                  # Multi-agent algorithms
    │   │   ├── imaxppo.py          # Our IMAX method
    │   │   └── mappo.py            # MAPPO method
    │   ├── policies
    │   │   └── policy.py           # Actor-critic policy
    │   └── utils.py
    ├── trainer                     
    │   ├── buffer.py               # Replay buffer
    │   ├── runner.py               # Trajectory collector
    │   └── trainer.py              # Multi-agent trainer
    ├── configs                     # Configuration
    ├── main.py                     
    └── utils.py                    

### Training our methods from scratch

- SMACv2: `python -u main.py --sc2-path=[SC2_PATH] --map-name [MAP_NAME]`
    - `[SC2_PATH]` is the directory of downloaded and extracted StarCraftII environment
    - `[MAP_NAME]` is name of SMACv2 scenario
- GRF: `python -u main.py --map-name [GRF_NAME] --n-timesteps=1000000 --ent-coef=0.0`
    - `[GRF_NAME]` is name of Google Research Fooball scenario
    - We supported `academy_3_vs_1_with_keeper`, `academy_counterattack_easy`, and `academy_counterattack_hard`
    - Other scenarios might be worked
- Miner: `python -u main.py --map-name [MINER_NAME] --n-envs=32`
    - `[MINER_NAME]` is name of Miner scenario
    - More details of this environmnet are listed in the folder [./miner](./miner)

### Evaluation and analysis

- To compare our method with existing approaches, please check this notebook [./benchmark.ipynb](./benchmark.ipynb) for analysis and visualization.
- Check our paper for more details on hyperparameter settings, configuration, and experimental results.

### Contact

- Due to double-blind peer review, our contact details will be revealed later.

