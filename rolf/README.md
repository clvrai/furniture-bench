# Robot Learning Framework for Research (RoLF)


## Reinforcement learning (RL) algorithms
* PPO
* DDPG
* TD3
* SAC
* Dreamer


## Imitation learning (IL) algorithms
* BC
* GAIL


## Directories
* `rolf/`:
  * `main.py`: sets up experiment and runs training using `trainer.py`
  * `trainer.py`: contains training and evaluation code
  * `algorithms/`: implementation of all RL and IL algorithms
  * `config/`: hyperparameters in yaml (using hydra)
    * `algo/`: hyperparameters for algorithms
    * `env/` : hyperparameters for environments
  * `networks/`: implementation of networks, such as policy and value function
  * `utils/`: contains helper functions


## Prerequisites
* Ubuntu 18.04 or above
* Python 3.9
* MuJoCo 2.1.0 and MuJoCo 2.1.1


## Installation

1. Install MuJoCo 2.1.0 and MuJoCo 2.1.1, and add the following environment variables into `~/.bashrc` or `~/.zshrc`
```bash
# download MuJoCo 2.1.0 for mujoco-py
$ mkdir ~/.mujoco
$ wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
$ tar -xvzf mujoco210_linux.tar.gz -C ~/.mujoco/
$ rm mujoco210_linux.tar.gz

# download MuJoCo 2.1.1 for dm_control
$ wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O mujoco211_linux.tar.gz
$ tar -xvzf mujoco211_linux.tar.gz -C ~/.mujoco/
$ rm mujoco211_linux.tar.gz

# add MuJoCo 2.1.0 to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

2. Install required dependencies
```bash
$ sudo apt-get install cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev

# software rendering
$ sudo apt-get install libgl1-mesa-glx libosmesa6 patchelf

# window rendering
$ sudo apt-get install libglfw3 libglew-dev
```

3. Install [appropriate version of PyTorch](https://pytorch.org/get-started/locally/)
```bash
# PyTorch 1.10.2, Linux, CUDA 11.3
$ pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

4. Finally, install `robot_learning (rolf)` package
```bash
# at the root directory (`robot_learning/`)
$ pip install -e .
```


## Usage

Use following commands to run RL/IL algorithms. Each experiment is represented as `[ENV].[ALGORITHM].[RUN_PREFIX].[SEED]` and checkpoints and videos are stored in `log/[ENV].[ALGORITHM].[RUN_PREFIX].[SEED]`. `run_prefix` can be used to differentiate runs with different hyperparameters. See `rolf/config/default_config.yaml` for the default hyperparameters.


### PPO
```bash
$ python -m rolf.main run_prefix=test algo@rolf=ppo env.id=Hopper-v2
```

### DDPG
```bash
$ python -m rolf.main run_prefix=test algo@rolf=ddpg env.id=Hopper-v2
```

### TD3
```bash
$ python -m rolf.main run_prefix=test algo@rolf=td3 env.id=Hopper-v2
```

### SAC
```bash
$ python -m rolf.main run_prefix=test algo@rolf=sac env.id=Hopper-v2
```

### BC
1. Generate demo using PPO
```bash
# train ppo expert agent
$ python -m rolf.main run_prefix=test algo@rolf=ppo env.id=Hopper-v2

# collect expert trajectories using ppo expert policy
$ python -m rolf.main run_prefix=test algo@rolf=ppo env.id=Hopper-v2 is_train=False record_video=False record_demo=True num_eval=100
# 100 trajectories are stored in log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

2. Run BC
```bash
$ python -m rolf.main run_prefix=test algo@rolf=bc env.id=Hopper-v2 demo_path=log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

### GAIL
```bash
$ python -m rolf.main run_prefix=test algo@rolf=gail env.id=Hopper-v2 demo_path=log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl

# GAIL with BC initialization
$ python -m rolf.main run_prefix=test algo@rolf=gail env.id=Hopper-v2 demo_path=log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl init_ckpt_path=log/Hopper-v2.bc.test.123/ckpt_00000020.pt init_ckpt_pretrained=True
```


## Implement your own algorithm
Implement your own `run.py` for experiment setup, `your_config.yaml` for configuration, `your_trainer.py` for training/evaluation loop, `your_agent.py` for algorithm, `your_rollout.py` for rollout, `your_network.py` for models.

Please refer to [`skill-chaining` repository](https://github.com/clvrai/skill-chaining) for an example. It implements `run.py` for experiment setup, `policy_sequencing_config.yaml` for configuration, `policy_sequencing_trainer.py` for training/evaluation loop, `policy_sequencing_agent.py` for algorithm, `policy_sequencing_rollout.py` for rollout.


## Papers using this code
* [Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization (CoRL 2021)](https://clvrai.com/skill-chaining)
* [Policy Transfer across Visual and Dynamics Domain Gaps via Iterative Grounding (RSS 2021)](https://clvrai.com/idapt)
* [IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks (ICRA 2021)](https://clvrai.com/furniture)
* [Motion Planner Augmented Reinforcement Learning for Robot Manipulation in Obstructed Environments (CoRL 2020)](https://clvrai.com/mopa-rl)
* [Learning to Coordinate Manipulation Skills via Skill Behavior Diversification (ICLR 2020)](https://clvrai.com/coordination)
