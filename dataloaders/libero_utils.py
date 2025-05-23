import os
import numpy as np
from sklearn.decomposition import PCA # type: ignore

from robomimic.utils.dataset import SequenceDataset # type: ignore
import robomimic.utils.obs_utils as ObsUtils # type: ignore
from libero.libero.benchmark import BENCHMARK_MAPPING # type: ignore


MODALITY = {
    'rgb': ["agentview_rgb", "eye_in_hand_rgb"],
    'depth': [],
    'low_dim': ["gripper_states", "joint_states"]
}
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": MODALITY})
BENCHMARK = BENCHMARK_MAPPING['libero_goal'](task_order_index=0)


def get_dataset(i, use_image_data, data_directory):
    obs_keys = ["gripper_states", "joint_states"]

    if use_image_data:
        obs_keys += ["agentview_rgb", "eye_in_hand_rgb"]

    dataset_path = os.path.join(
        data_directory, BENCHMARK.get_task_demonstration(i))
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=["actions", "dones"],
        load_next_obs=False,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,  
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode='low_dim', 
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
    )
    return dataset


def adapt_data_to_bet(
        dataset,
        horizon = 400,
        subtask=0,
        obs_dim=None,
        act_dim=None,
        use_image_data=False):
    n_demos = dataset.n_demos
    observations = np.zeros((n_demos, horizon, obs_dim))
    actions = np.zeros((n_demos, horizon, act_dim))
    masks = np.zeros((n_demos, horizon))
    gt_options = np.ones((n_demos, horizon)) * subtask
    imgs1 = np.zeros((n_demos, horizon, 3, 128, 128), dtype=np.uint8)
    imgs2 = np.zeros((n_demos, horizon, 3, 128, 128), dtype=np.uint8)
    curr_demo = 0
    prev_steps = 0
    for idx in range(len(dataset)):
        data = dataset[idx]
        concat = np.concatenate((data['obs']['gripper_states'], 
                                 data['obs']['joint_states']), axis=1)        
        observations[curr_demo][idx-prev_steps] = concat
        actions[curr_demo][idx-prev_steps] = data['actions']
        masks[curr_demo][idx-prev_steps] = 1
        if use_image_data:
            imgs1[curr_demo][idx-prev_steps] = np.moveaxis(
                data['obs']['agentview_rgb'][0], 2, 0)
            imgs2[curr_demo][idx-prev_steps] = np.moveaxis(
                data['obs']['eye_in_hand_rgb'][0], 2, 0)

        if data['dones']==1:
            curr_demo +=1
            prev_steps = idx+1
    if use_image_data:
        return observations, actions, masks, gt_options, imgs1, imgs2
    
    else:
        return observations, actions, masks, gt_options
