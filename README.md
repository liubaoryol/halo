# Hierarchical Active Learning for Options (HALO) 
Hierarchical Reinforcement Learning (HRL) enables solving complex, long-horizon tasks by decomposing them into meaningful subtasks or options.  While the benefits of hierarchical structures are well-established, extending these approaches to imitation learning introduces a key challenge: the need to annotate options in expert trajectories, substantially increasing the annotation burden. Current Hierarchical Imitation Learning (HIL) methods either rely on exhaustive option annotation or attempt unsupervised option discovery, which often fails to capture semantically meaningful decompositions. We introduce Hierarchical Active Learning of Options (HALO), an active imitation learning algorithm that efficiently learns hierarchical policies by strategically querying for option labels at the most informative timesteps. We provide theoretical bounds on query efficiency and demonstrate that HALO consistently outperforms both unsupervised option discovery methods and standard imitation learning, achieving comparable performance to fully supervised HIL with only a small fraction of option annotations. Experiments across grid environments and robotic manipulation tasks show that our method achieves superior task performance while learning interpretable option boundaries that align with semantic task structure.

## ![status](https://img.shields.io/badge/status-beta-yellow) Code release


This software can be used to replicate the experiments from the associated paper. However, it is still under active development, and future updates will improve standardization and readability.

The following instructions assume the current working directory is the root of this repository. The code has been tested on Ubuntu 20.04 LTS (amd64).

## Getting Started

### Project Setup

First, install system dependencies:

```bash
sudo apt update
sudo apt install -y build-essential libgl1
```
Then, clone the repository and set up the conda environment:
```bash
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

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..
```

To enable logging with Weights & Biases (wandb), log in with a `wandb` account:
```bash
wandb login
```
To disable logging entirely (e.g. for offline use):
```bash
export WANDB_MODE=disabled
```

## Setting Up Domains and Downloading Training Datasets

After downloading the datasets, you need to update `configs/env_vars/env_vars.yaml` to point to the correct local paths.

If you're following the setup as described here, following the command prompts, and you're in the `halo` directory, the following command will automatically update the paths:

```bash
sed -i "s|/tmp/Documents/my-packages/|$(pwd)/|g" configs/env_vars/env_vars.yaml
```

All datasets below will be saved in a `datasets/` folder created inside your current `halo` directory.


🧭 **Rescue-World Dataset**: This dataset contains three configurations with `n = 2, 3, 6` medical kits, as described in the paper. It can be found [here](https://osf.io/54xah/files/osfstorage). To download them via command line:

```bash
mkdir -p datasets

wget [redacted-double-blind-review] -O datasets/rescue_world_n2.tar.gz && tar -xzvf datasets/rescue_world_n2.tar.gz -C datasets
wget [redacted-double-blind-review] -O datasets/rescue_world_n3.tar.gz && tar -xzvf datasets/rescue_world_n3.tar.gz -C datasets
wget [redacted-double-blind-review] -O datasets/rescue_world_n6.tar.gz && tar -xzvf datasets/rescue_world_n6.tar.gz -C datasets
```
🤖 **Franka Kitchen Dataset**

MuJoCo 2.1.0 is only required for evaluation, as the training does not require interaction with environment (offline imitation learning)

- Install MuJoCo 2.1.0: [Installation Guide](https://github.com/openai/mujoco-py#install-mujoco)

Download the dataset:

```bash
wget https://osf.io/download/4g53p -O datasets/bet_data_release.tar.gz && tar -xzvf datasets/bet_data_release.tar.gz -C datasets
```

🧩 **LIBERO Dataset**: To download the `libero_goal` dataset, use the author's code directly. For more detail refer to their [GitHub page](https://github.com/Lifelong-Robot-Learning/LIBERO) or [project page](https://libero-project.github.io/datasets):

```bash
mkdir -p datasets/libero
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal --save_dir datasets/libero
```

## Training Commands

Training progress can be viewed in Weights & Biases (wandb).  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_kitchen_train`

All commands have the same options, where you can choose `student_type` from 
- halo
- uniform
- supervised
- no_query

```bash
# Rescue World
python3 train_hbc.py --config-name=train_rw_n2 student_type=halo seed=0 query_percentage_budget=0.3
# Franka Kitchen
python3 train.py --config-name=train_kitchen seed=6 project=neurips2025_kitchen student_type=latent_entropy_based query_percentage_budget=0.2
# LIBERO
python3 train.py --config-name=train_libero student_type=latent_entropy_based seed=0 project=exp1

## Evaluation
**Rescue World**: Training script also performs evaluation, but here is a minimal example to evaluate HBC on rescue world gym environment
```python
from evaluate import evaluate_policy
from models.hbc import HBC

hbc = HBC(...)  # Load your trained model
mean_return, std_return = evaluate_policy(hbc, env, n_eval_episodes=10)
```
**Franka Kitchen**
1. In configs/env/relay_kitchen_traj.yaml, set load_dir to the absolute path of the directory containing the trained model.
2. Evaluation requires including the Relay Policy Learning repository in PYTHONPATH. `export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs`
3. `python3 run_on_env.py --config-name=eval_kitchen env.load_dir=$(pwd)/exp_remote/kitchen/2025.04.08/030424_kitchen_train`

To speed up evaluation, rendering can be disabled for the kitchen environment by setting the following in configs/eval_kitchen.yaml: `enable_render: False`
#### LIBERO
` python3 run_on_env.py --config-name=eval_libero env.load_dir=$(pwd)/exp_local/2025.03.23/203929_libero_train`


## Acknowledgements

- [notmahi/bet](https://github.com/notmahi/bet): for releasing behavior training transformers. HALO was buit by extending BeT to the hierarchical version.
- [libero](https://github.com/Lifelong-Robot-Learning/LIBERO) for their dataset, which was essential to test query functions
- [compile](https://arxiv.org/abs/1812.01483) for releasing the minimal code to implement compositional imitation learning.
- [facebookresearch/hydra](https://github.com/facebookresearch/hydra): Used for flexible configuration management.

