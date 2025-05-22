# Hierarchical Active Learning for Options (HALO) 
Hierarchical Reinforcement Learning (HRL) enables solving complex, long-horizon tasks by decomposing them into meaningful subtasks or options.  While the benefits of hierarchical structures are well-established, extending these approaches to imitation learning introduces a key challenge: the need to annotate options in expert trajectories, substantially increasing the annotation burden. Current Hierarchical Imitation Learning (HIL) methods either rely on exhaustive option annotation or attempt unsupervised option discovery, which often fails to capture semantically meaningful decompositions. We introduce Hierarchical Active Learning of Options (HALO), an active imitation learning algorithm that efficiently learns hierarchical policies by strategically querying for option labels at the most informative timesteps. We provide theoretical bounds on query efficiency and demonstrate that HALO consistently outperforms both unsupervised option discovery methods and standard imitation learning, achieving comparable performance to fully supervised HIL with only a small fraction of option annotations. Experiments across grid environments and robotic manipulation tasks show that our method achieves superior task performance while learning interpretable option boundaries that align with semantic task structure.

## Code release
![status](https://img.shields.io/badge/status-beta-yellow)

This software can be used to replicate the experiments from the associated paper. However, it is still under active development, and future updates will improve standardization and readability.

The following instructions assume the current working directory is the root of this repository. The code has been tested on Ubuntu 20.04 LTS (amd64).

## Getting Started

### 🔧 Project Setup

First, install system dependencies:

```bash
sudo apt update
sudo apt install -y build-essential libgl1
```
Then, clone the repository and set up the conda environment:
```
git clone https://github.com/liubaoryol/halo.git
cd halo

# Initialize Conda (only needed once)
conda init
source ~/.bashrc

# Create and activate the conda environment
conda create --name halo --file spec.txt
conda activate halo

# Install Python dependencies
pip install -r requirements.txt
pip install torch wandb
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO.git):

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
```

To enable logging with Weights & Biases (wandb), log in with a `wandb` account:
```
wandb login
```
To disable logging entirely (e.g. for offline use):
```
export WANDB_MODE=disabled
```

## Setting Up Domains and Downloading Training Datasets

After downloading the datasets, you need to update `configs/env_vars/env_vars.yaml` to point to the correct local paths.

If you're following the setup as described here, following the command prompts, and you're in the `halo` directory, the following command will automatically update the paths:

```bash
sed -i "s|/home/liubove/Documents/my-packages/|$(pwd)/|g" configs/env_vars/env_vars.yaml
```

All datasets below will be saved in a `datasets/` folder created inside your current `halo` directory.


### 🧭 Rescue-World Dataset

This dataset contains three configurations with `n = 2, 3, 6` medical kits, as described in the paper. It can be found [here](https://osf.io/54xah/files/osfstorage)

To download them via command line:

```bash
mkdir -p datasets

wget https://osf.io/a3t8y/download -O datasets/rescue_world_n2.tar.gz && tar -xzvf datasets/rescue_world_n2.tar.gz -C datasets
wget https://osf.io/y9z67/download -O datasets/rescue_world_n3.tar.gz && tar -xzvf datasets/rescue_world_n3.tar.gz -C datasets
wget https://osf.io/92sg6/download -O datasets/rescue_world_n6.tar.gz && tar -xzvf datasets/rescue_world_n6.tar.gz -C datasets
```


### 🤖 Franka Kitchen Dataset

**MuJoCo 2.1.0** is only required for **evaluation**, as the training does not require interaction with environment (offline imitation learning)

- Install MuJoCo 2.1.0: [Installation Guide](https://github.com/openai/mujoco-py#install-mujoco)

Download the dataset:

```bash
wget https://osf.io/download/4g53p -O datasets/bet_data_release.tar.gz && tar -xzvf datasets/bet_data_release.tar.gz -C datasets
```



### 🧩 LIBERO Dataset

To download the `libero_goal` dataset, use the author's code directly. For more detail refer to their 
[github page](https://github.com/Lifelong-Robot-Learning/LIBERO or [project page](https://libero-project.github.io/datasets):

```bash
mkdir -p datasets/libero
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal --save_dir datasets/libero
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
Code build on top of [Behavior Transformers (BET)](https://github.com/notmahi/bet) github to include expert querying. Besides BeT, also Behavioral Cloning model is used.
Some code from to test Compile is also borrowed.

- [facebookresearch/hydra](https://github.com/facebookresearch/hydra): Configuration managements.
- [libero]()
  
