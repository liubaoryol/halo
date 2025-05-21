import os
import sys
sys.path.append('/home/liubove/Documents/git-packages/TinySAM/')
import pickle
import numpy as np
from sklearn.decomposition import PCA # type: ignore
import torch # type: ignore

# from tinysam import sam_model_registry, SamPredictor # type: ignore
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
DATA_DIRECTORY = '/home/liubove/Documents/my-packages/LIBERO/libero/datasets'
TINYSAM ="/home/liubove/Documents/git-packages/TinySAM/weights/tinysam_42.3.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset(i, use_image_data, data_directory=DATA_DIRECTORY):
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


# def set_sam_encoder():
#     sam = sam_model_registry['vit_t'](checkpoint=TINYSAM)
#     sam.to(device=DEVICE)
#     predictor = SamPredictor(sam)
#     return predictor


def get_libero_images_feats(dataset, predictor, pca_agentview, pca_eye_in_hand):
    agent_images = []
    eye_in_hand_images = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        img1 = data['obs']['agentview_rgb']
        img2 = data['obs']['eye_in_hand_rgb']

        with torch.no_grad():
            predictor.set_image(img1[0])
        features1 = predictor.features.reshape(1,-1).cpu().numpy()
        features1 = pca_agentview.transform(features1)
        agent_images.append(features1)
        
        with torch.no_grad():
            predictor.set_image(img2[0])
        features2 = predictor.features.reshape(1,-1).cpu().numpy()
        features2 = pca_eye_in_hand.transform(features2)
        eye_in_hand_images.append(features2)
    
    return agent_images, eye_in_hand_images


def get_libero_images_feats_all(datasets, predictor, pca_agentview, pca_eye_in_hand):
    agent_images_full = []
    eye_in_hand_images_full = []
    for dataset in datasets:
        print("Getting features from ", dataset.hdf5_path)
        agent_images, eye_in_hand_images = get_libero_images_feats(dataset,
                                                             predictor,
                                                             pca_agentview,
                                                             pca_eye_in_hand
                                                             )
        agent_images_full += agent_images
        eye_in_hand_images_full += eye_in_hand_images

    return agent_images_full, eye_in_hand_images_full

import torchvision

def get_libero_images(dataset):
    agent_images = []
    eye_in_hand_images = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        img1 = data['obs']['agentview_rgb']
        img2 = data['obs']['eye_in_hand_rgb']
        agent_images.append(img1)
        eye_in_hand_images.append(img2)
    
    return agent_images, eye_in_hand_images


def get_libero_images_all(datasets):
    agent_images_full = []
    eye_in_hand_images_full = []
    for dataset in datasets:
        print("Getting images from ", dataset.hdf5_path)
        agent_images, eye_in_hand_images = get_libero_images(dataset)
        agent_images_full += agent_images
        eye_in_hand_images_full += eye_in_hand_images

    return agent_images_full, eye_in_hand_images_full


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

def append_images_to_robot_state(observations, image_features, masks):
    n_demos, horizon, obs_dim = observations.shape
    obs_dim += image_features.shape[1]
    new_observations = np.zeros((n_demos, horizon, obs_dim))
    counter = 0
    for traj, mask in enumerate(masks):
        for idx, m in enumerate(mask):
            if m:
                new_observations[traj, idx] = np.concatenate(
                    (observations[traj, idx], image_features[counter]))
                counter += 1
            else:
                break
    return new_observations

def fit_pca_models(imgs1, imgs2):
    pca_agentview = PCA(n_components=64)
    pca_eye_in_hand = PCA(n_components=64)
    pca_agentview.fit(np.vstack(imgs1))
    pca_eye_in_hand.fit(np.vstack(imgs2))
    pickle.dump(pca_agentview, open("pca_agentview.pkl","wb"))
    pickle.dump(pca_eye_in_hand, open("pca_eye_in_hand.pkl","wb"))
    return pca_agentview, pca_eye_in_hand


# predictor = set_sam_encoder()
# pca_agentview = pickle.load(open("pca_agentview.pkl",'rb')) 
# pca_eye_in_hand = pickle.load(open("pca_eye_in_hand.pkl",'rb'))

# datasets = [get_dataset(i, True, DATA_DIRECTORY) for i in range(10)]
# imgs1, imgs2 = get_libero_images_feats_all(datasets, predictor, pca_agentview, pca_eye_in_hand)

# agentview_feats = np.vstack(imgs1)
# eye_in_hand_feats = np.vstack(imgs2)
# np.save('agentview_feats', agentview_feats)
# np.save('eye_in_hand_feats', eye_in_hand_feats)