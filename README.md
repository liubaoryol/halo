# Hierarchical Active Learning for Options (HALO)


Code build on top of [Behavior Transformers (BET)](https://github.com/notmahi/bet) github to include expert querying. Besides BeT, also Behavioral Cloning model is used.
Some code from to test Compile is also borrowed.



## Abstract
Hierarchical Reinforcement Learning (HRL) enables solving complex, long-horizon tasks by decomposing them into meaningful subtasks or options.  While the benefits of hierarchical structures are well-established, extending these approaches to imitation learning introduces a key challenge: the need to annotate options in expert trajectories, substantially increasing the annotation burden. Current Hierarchical Imitation Learning (HIL) methods either rely on exhaustive option annotation or attempt unsupervised option discovery, which often fails to capture semantically meaningful decompositions. We introduce Hierarchical Active Learning of Options (HALO), an active imitation learning algorithm that efficiently learns hierarchical policies by strategically querying for option labels at the most informative timesteps. We provide theoretical bounds on query efficiency and demonstrate that HALO consistently outperforms both unsupervised option discovery methods and standard imitation learning, achieving comparable performance to fully supervised HIL with only a small fraction of option annotations. Experiments across grid environments and robotic manipulation tasks show that our method achieves superior task performance while learning interpretable option boundaries that align with semantic task structure.

## Code release

In this repository, you can find the code to reproduce Behavior Transformer (BeT). The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).

## Getting started
### Setting up the project
```
sudo apt update
sudo apt install build-essential
sudo apt install -y libgl1
git clone -b information-gain https://github.com/liubaoryol/bet.git
cd bet
conda init
source ~/.bashrc
conda create --name behavior-transformer --file spec.txt
conda activate behavior-transformer
pip install -r requirements.txt
pip install torch
pip install wandb
cd ..
sed -i "s|/home/liubove/Documents/my-packages/|$(pwd)/|g" bet/configs/env_vars/env_vars.yaml
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
```

### Logging with wandb

To enable logging, log in with a `wandb` account:
```
wandb login
```
Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
```
export WANDB_MODE=disabled
```


### Setting up domains and getting the training datasets
#### RW4T

#### Franka Kitchen
- Install MuJoCo 2.1.0: https://github.com/openai/mujoco-py#install-mujoco
- Download dataset
```
cd bet
wget https://osf.io/download/4g53p -O bet_data_release.tar.gz && tar -xzvf bet_data_release.tar.gz
```
#### LIBERO
- Download dataset
```
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal
```

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository.

To reproduce the experiment results, the overall steps are:
1. Activate the conda environment with
   ```
   conda activate behavior-transformer
   ```
2. Train with `python3 train.py`. A model snapshot will be saved to `./exp_local/...`;
  - use student_type arg to set student type.
    student types are:
    - latent_entropy_based
    - iterative_random
    - supervised
    - unsupervised
    - random
3. In the corresponding environment config, set the `load_dir` to the absolute path of the snapshot directory above;
4. Eval with `python3 run_on_env.py`.

See below for detailed steps for each environment.

### RW4T

### Franka kitchen

- Train:
  ```
  python3 train.py --config-name=train_kitchen seed=6 project=neurips2025_kitchen batch_size=64 student_type=latent_entropy_based query_percentage_budget=0.2
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_kitchen_train`
- In `configs/env/relay_kitchen_traj.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs
  python3 run_on_env.py --config-name=eval_kitchen env.load_dir=$(pwd)/exp_remote/kitchen/2025.04.08/030424_kitchen_train
  2025.04.08/030341_kitchen_train
  ```
  (Evaluation requires including the relay policy learning repo in `PYTHONPATH`.)


### Speeding up evaluation
- Rendering can be disabled for the kitchen environment: set `enable_render: False` in `configs/eval_kitchen.yaml`

### LIBERO
- Train:
  ```
  python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=0 project=exp1 batch_size=64
  ```
- Evaluation:
  ```
  python3 run_on_env.py --config-name=eval_libero env.load_dir=/home/liubove/Documents/my-packages/bet/exp_local/2025.03.23/203929_libero_train
  ```


## Acknowledgements

- [notmahi/bet](https://github.com/notmahi/bet): 
- [facebookresearch/hydra](https://github.com/facebookresearch/hydra): Configuration managements.
- [libero]()
