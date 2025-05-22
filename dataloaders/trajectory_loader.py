import logging
import os
import time
from typing import Union, Callable, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from utils import (
    transpose_batch_timestep,
    split_datasets,
)


from students.base import Oracle
from dataloaders.latent_estimation.parallelize_latent_estimation import (
    paralellize_prob_latent,
    single_prob_latent)
from dataloaders.libero_utils import get_dataset, adapt_data_to_bet


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

ALL_TASKS = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]


class RelayKitchenTrajectoryDataset(TensorDataset):
    def __init__(self,
                 data_directory='',
                 device="cpu",
                 obs_shape=60):
        data_directory = Path(data_directory)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks = transpose_batch_timestep(
            observations, actions, masks
        )
        masks = torch.from_numpy(masks).to(device).float()
        self.masks = masks
        self.device = device
        # available_opts
        gt_options = self._calculate_gt_options(observations)
        self.oracle = Oracle(true_options=gt_options)
        observations = observations[:,:,:obs_shape]
        
        self.options = torch.zeros_like(masks).int()
        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
            masks,
            self.options,
            torch.from_numpy(gt_options).to(device).int()

        )
        self.actions = self.tensors[1]

    def query_oracle(self, student, num_queries=1):
        return student.query_oracle(self.oracle, num_queries)

    def update_options(self, student):
        opts = self.get_probs(student)
        self.latent_probs = opts
        for i, opt in enumerate(opts):
            opt = torch.multinomial(torch.from_numpy(opt), 1)
            self.options[i][:len(opt)] = opt.squeeze(1)
        obs, acts, masks, _, gt_opts = self.tensors
        self.tensors = (obs, acts, masks, self.options, gt_opts)

    def update_options_of_traj(
            self,
            student,
            traj_num
            ):
        observations, actions, masks, _, gt_opts = self.tensors
        obs = observations[traj_num].unsqueeze(0)
        acts = actions[traj_num].unsqueeze(0)
        from dataloaders.latent_estimation.log_probs import aux_probs
        prob_acts, prob_opts = aux_probs(
            obs.to('cuda'),
            acts.to('cuda'),
            student.state_prior,
            student.action_ae,
            option_dim=7
            )
        last_step = np.where(self.masks[traj_num]==0)[0]
        if len(last_step)>0:
            last_step=last_step[0].item()
        else:
            last_step=len(obs[0])
        args= (traj_num,
                    prob_acts[0].cpu().numpy(),
                    prob_opts[0].cpu().numpy(),
                    last_step,
                    student.list_queries,
                    student.annotated_options,
                    7)
        self.latent_probs[traj_num] = single_prob_latent(args)
        opts = self.latent_probs[traj_num]
        try:
            opts = torch.multinomial(torch.from_numpy(opts), 1)
        except:
            import pdb; pdb.set_trace()
        self.options[traj_num][:len(opts)] = opts.squeeze(1)

    def get_probs(self, student, save=True):
        print("Estimating probs for all traj latents. Please wait...")
        now = time.perf_counter()
        observations, actions, masks, _, _ = self.tensors
        probs = paralellize_prob_latent(observations.to('cuda'),
                                        actions.to('cuda'),
                                        masks.to('cuda'),
                                        student)
        transcurrido = time.perf_counter()-now
        print("Done! Elapsed time for ", len(self), " trajectories was: ", transcurrido)
        if save:
            self.latent_probs=probs
        return probs

    def _calculate_gt_options(self, observations):
        true_options = []
        for episode in observations:
            args = []
            for GOAL in ALL_TASKS:
                obj_state = episode[:, OBS_ELEMENT_INDICES[GOAL]]
                obj_goal = OBS_ELEMENT_GOALS[GOAL]
                arg = np.ceil(np.linalg.norm(obj_state - obj_goal, axis=1)*100).argmin()
                args.append(arg)

            opts = np.zeros(len(episode), int)
            sorted_args = np.argsort(args)
            opts[:args[sorted_args[0]]] = sorted_args[0]
            for idx, (arg1, arg2) in enumerate(zip(sorted_args[:-1], sorted_args[1:])):
                opts[args[arg1]:args[arg2]] = arg2
            # opts[args[arg2]:] = idx+1

            true_options.append(opts)

        return np.stack(true_options)
    
    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class LiberoTrajectoryDataset(TensorDataset):
    def __init__(self,
                 data_directory=None,
                 use_image_data=False,
                 device="cpu"):

        observations = []
        actions = []
        masks = []
        self.use_image_data = use_image_data
        gt_options = []
        self.datasets = []
        for i in range(10):  
            dataset = get_dataset(i, use_image_data, data_directory)
            obs, acts, mask, gt_opts = adapt_data_to_bet(
                dataset,
                subtask=i,
                obs_dim = 9,
                act_dim=7)
                
            observations.append(obs)
            actions.append(acts)
            masks.append(mask)
            gt_options.append(gt_opts)
            self.datasets.append(dataset)

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        masks = np.vstack(masks)
        gt_options = np.vstack(gt_options)

        masks = torch.from_numpy(masks).to(device).float()
        self.masks = masks
        self.device = device
        self.oracle = Oracle(true_options=gt_options)
        self.n_trjs, self.len_trjs = masks.shape
        self.options = torch.zeros_like(masks).int()
        self.set_idx2trj_dict()
        if use_image_data:
            super().__init__(
                torch.from_numpy(observations).to(device).float(),
                torch.from_numpy(actions).to(device).float(),
                masks,
                self.options,
                torch.from_numpy(gt_options).to(device).int(),
                torch.arange(self.n_trjs*self.len_trjs).reshape(self.n_trjs, self.len_trjs)
            )
        else:
            super().__init__(
                torch.from_numpy(observations).to(device).float(),
                torch.from_numpy(actions).to(device).float(),
                masks,
                self.options,
                torch.from_numpy(gt_options).to(device).int(),
            )
        self.actions = self.tensors[1]

    # def __len__(self):
    #     return int(self.masks.sum().item())

    def get_images(self, idx):
        dataset_num, timestep = self.idx2trj_dict[idx]
        dataset = self.datasets[dataset_num]
        obs = dataset[timestep]['obs']

        agentview_rgb = np.moveaxis(obs['agentview_rgb'][0], 2, 0)
        eye_in_hand_rgb = np.moveaxis(obs['eye_in_hand_rgb'][0], 2, 0)
        return agentview_rgb, eye_in_hand_rgb

    def set_idx2trj_dict(self):
        self.idx2trj_dict = {}
        for dataset_num, dataset in enumerate(self.datasets):
            trj_num = 0
            timestep = 0
            for data_idx in range(len(dataset)):
                data = dataset[data_idx]
                done = data['dones'].item()
                idx = dataset_num * 50 *400 + trj_num * 400 + timestep
                self.idx2trj_dict[idx] = (dataset_num, data_idx)
                timestep +=1
                if done:
                    trj_num +=1
                    timestep = 0

    def get_stats(self):
        stats = {t:0 for t in range(10)}
        for data in self:
            state_action_pairs = data[2].sum()
            t = data[4][0].item()
            stats[t] += state_action_pairs
        print(stats)

    def query_oracle(self, student, num_queries=1):
        return student.query_oracle(self.oracle, num_queries)

    def update_options(self, student):
        opts = self.get_probs(student)
        self.latent_probs = opts
        for i, opt in enumerate(opts):
            opt = torch.multinomial(torch.from_numpy(opt), 1)
            self.options[i][:len(opt)] = opt.squeeze(1)
        if self.use_image_data:
            obs, acts, masks, _, gt_opts, idx = self.tensors
            self.tensors = (obs, acts, masks, self.options, gt_opts, idx)
        else:
            obs, acts, masks, _, gt_opts = self.tensors
            self.tensors = (obs, acts, masks, self.options, gt_opts)

    def update_options_of_traj(
            self,
            student,
            traj_num
            ):
        
        if self.use_image_data:
            observations, actions, masks, _, gt_opts, idx = self.tensors
        else:
            observations, actions, masks, _, gt_opts = self.tensors
        
        obs = observations[traj_num].unsqueeze(0)
        acts = actions[traj_num].unsqueeze(0)
        from dataloaders.latent_estimation.log_probs import aux_probs
        prob_acts, prob_opts = aux_probs(
            obs.to('cuda'),
            acts.to('cuda'),
            student.state_prior,
            student.action_ae,
            option_dim=10
            )
        last_step = np.where(self.masks[traj_num]==0)[0]
        if len(last_step)>0:
            last_step=last_step[0].item()
        else:
            last_step=len(obs[0])
        args= (traj_num,
                    prob_acts[0].cpu().numpy(),
                    prob_opts[0].cpu().numpy(),
                    last_step,
                    student.list_queries,
                    student.annotated_options,
                    10)
        self.latent_probs[traj_num] = single_prob_latent(args)
        opts = self.latent_probs[traj_num]
        try:
            opts = torch.multinomial(torch.from_numpy(opts), 1)
        except:
            import pdb; pdb.set_trace()
        self.options[traj_num][:len(opts)] = opts.squeeze(1)

    def get_probs(self, student, save=True):
        print("Estimating probs for all traj latents. Please wait...")
        now = time.perf_counter()
        if self.use_image_data:
            observations, actions, masks, _, _, idxs = self.tensors
        else:
            observations, actions, masks, _, _ = self.tensors
            idxs = None
        
        probs = paralellize_prob_latent(observations.to('cuda'),
                                        actions.to('cuda'),
                                        masks.to('cuda'),
                                        student,
                                        option_dim=10,
                                        idxs=idxs,
                                        dataset=self)
        transcurrido = time.perf_counter()-now
        print("Done! Elapsed time for ", len(self), " trajectories was: ", transcurrido)
        if save:
            self.latent_probs=probs
        return probs

    def _calculate_gt_options(self, observations):
        true_options = []
        for episode in observations:
            args = []
            for GOAL in ALL_TASKS:
                obj_state = episode[:, OBS_ELEMENT_INDICES[GOAL]]
                obj_goal = OBS_ELEMENT_GOALS[GOAL]
                arg = np.ceil(np.linalg.norm(obj_state - obj_goal, axis=1)*100).argmin()
                args.append(arg)

            opts = np.zeros(len(episode), int)
            sorted_args = np.argsort(args)
            opts[:args[sorted_args[0]]] = sorted_args[0]
            for idx, (arg1, arg2) in enumerate(zip(sorted_args[:-1], sorted_args[1:])):
                opts[args[arg1]:args[arg2]] = arg2
            # opts[args[arg2]:] = idx+1

            true_options.append(opts)

        return np.stack(true_options)
    
    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class TrajectorySlicerDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        window: int,
        transform: Optional[Callable] = None,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.

        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        returns: a dataset of sequences of length `window`
        """
        self.dataset = dataset
        self.window = window
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):  # type: ignore
            T = self._get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window)
                ]  # slice indices follow convention [start, end)

            if min_seq_length < window:
                print(
                    f"Ignored short sequences. To include all, set window <= {min_seq_length}."
                )

    def _get_seq_length(self, idx: int) -> int:
        # Adding this convenience method to avoid reading the actual sequence
        # We retrieve the length in trajectory slicer just so we can use subsetting
        # and shuffling before we pass a dataset into TrajectorySlicerDataset
        return self.dataset.get_seq_length(idx)

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        values = tuple(
            x[start:end] for x in self.dataset[i]
        )  # (observations, actions, mask)
        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return values


class TrajectorySlicerSubset(TrajectorySlicerDataset):
    def _get_seq_length(self, idx: int) -> int:
        # self.dataset is a torch.dataset.Subset, so we need to use the parent dataset
        # to extract the true seq length.
        subset = self.dataset
        return subset.dataset.get_seq_length(subset.indices[idx])  # type: ignore

    def _get_all_actions(self) -> torch.Tensor:
        return self.dataset.dataset.get_all_actions()


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    use_image_data=None
):

    relay_kitchen_trajectories = RelayKitchenTrajectoryDataset(
        data_directory)
    
    train_set, val_set = split_datasets(
        relay_kitchen_trajectories,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    # Convert to trajectory slices.
    train_trajectories = TrajectorySlicerSubset(train_set, window=window_size)
    val_trajectories = TrajectorySlicerSubset(val_set, window=window_size)
    return train_trajectories, val_trajectories


def get_libero_train_val(
        data_directory,
        train_fraction=0.9,
        random_seed=42,
        device="cpu",
        window_size=10,
        use_image_data=False
):
    dataset_full = LiberoTrajectoryDataset(data_directory, use_image_data)
    train_set, val_set = split_datasets(
        dataset_full,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_trajectories = TrajectorySlicerSubset(train_set, window=window_size)
    val_trajectories = TrajectorySlicerSubset(val_set, window=window_size)
    return train_trajectories, val_trajectories


def get_rescue_world_data(data_directory):
    gini = Oracle.load(data_directory)
    return gini

