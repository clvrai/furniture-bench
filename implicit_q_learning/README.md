# Offline Reinforcement Learning with Implicit Q-Learning

## Change for FurnitureBenchmark

### Need to fix GYM

Comment out the following code in `gym/utils/passive_env_checker.py:passive_env_step_check()`:
```
    # if np.any(np.isnan(obs)):
    #     logger.warn("Encountered NaN value in observations.")
    # if np.any(np.isinf(obs)):
    #     logger.warn("Encountered inf value in observations.")
```


### Data
```bash
$ python convert_furniture_data.py --furniture={furniture} --demo_dir={data/dir/path}
# e.g.
$ python convert_furniture_data.py --furniture=drawer --demo_dir=drawer_image
```

Script above will convert demonstrations to a pickle file that can be used by IQL code. Converted demonstrations will be stored in `data/Image/FURNITURE.pkl` or `data/Parts/FURNITURE.pkl`.

### Training
`--run_name`: specify run name. Log dir name is `RUN_NAME.SEED`.
`--env_name=ENV_ID/FURNITURE`
`--config`: we need to find `expectile`, `temperature`, and `dropout_rate`.

```bash
$ python train_offline.py --env_name=Furniture-IQL-Dummy-v0/square_table --config=configs/furniture_config.py --run_name debug --data_path={/path/to/data}
```

### Testing
`--run_name`: specify run name you used for the training. Log dir name is `RUN_NAME.SEED`. This will determine the ckpt you want to load.
`--ckpt_step`: specify ckpt number you want to load.
```bash
# E.g.
$ python test_offline.py --env_name=Furniture-IQL-v0/square_table --config=configs/furniture_config.py --ckpt_step=1000000 --run_name debug
```


## Original README

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

For a PyTorch reimplementation see https://github.com/rail-berkeley/rlkit/tree/master/examples/iql

## How to run the code

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install --upgrade "jax[cuda]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
