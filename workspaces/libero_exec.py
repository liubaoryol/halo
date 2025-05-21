import einops
import gym
import hydra
import joblib
import torch
import umap
import umap.plot
import wandb
import logging
import pickle

import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from workspaces import base
import envs
# from dataloaders.libero_utils import set_sam_encoder


# PREDICTOR = set_sam_encoder()
# pca_agentview = pickle.load(open("/home/liubove/Documents/my-packages/bet/pca_agentview.pkl",'rb')) 
# pca_eye_in_hand = pickle.load(open("/home/liubove/Documents/my-packages/bet/pca_eye_in_hand.pkl",'rb'))

class LiberoWorkspace(base.Workspace):
    def _setup_plots(self):
        plt.ion()
        obs_mapper_path = (
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "obs_mapper.pkl"
        )
        with (obs_mapper_path).open("rb") as f:
            obs_mapper = joblib.load(f)
        self.obs_mapper = obs_mapper
        # self.obs_ax = umap.plot.points(obs_mapper)
        self.obs_ax = plt.scatter(
            obs_mapper.embedding_[:, 0], obs_mapper.embedding_[:, 1], s=0.01, alpha=0.1
        )
        self.obs_sc = plt.scatter([0], [0], marker="X", c="orange")
        self._figure_1 = plt.gcf()

        self._figure_2 = plt.figure()
        action_mapper_path = (
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "action_mapper.pkl"
        )
        with (action_mapper_path).open("rb") as f:
            action_mapper = joblib.load(f)
        self.action_mapper = action_mapper
        # self.action_ax = umap.plot.points(action_mapper)
        self.action_ax = plt.scatter(
            action_mapper.embedding_[:, 0],
            action_mapper.embedding_[:, 1],
            s=0.01,
            alpha=0.1,
        )
        self.action_sc = plt.scatter([0], [0], marker=".", c="orange", alpha=0.5)
        plt.draw()
        plt.pause(0.001)

    def _setup_starting_state(self):
        self.init_qpos = np.load(
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "all_init_qpos.npy"
        )
        self.init_qvel = np.load(
            Path(self.cfg.env_vars.datasets.relay_kitchen) / "all_init_qvel.npy"
        )

    def _start_from_known(self):
        return self.env.reset_from_known()

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        obs_embedding = self.obs_mapper.transform(
            einops.rearrange(obs, "(batch obs) -> batch obs", batch=1)
        )
        self.obs_sc.set_offsets(obs_embedding[:, :2])
        self.obs_sc.set_sizes([50])

        expanded_chosen_action = einops.rearrange(
            chosen_action, "(batch obs) -> batch obs", batch=1
        )
        action_embedding = self.action_mapper.transform(expanded_chosen_action)
        colors = ["orange"]
        sizes = [50]
        if all_actions is not None:
            all_action_embedding = self.action_mapper.transform(all_actions)
            action_embedding = np.concatenate([action_embedding, all_action_embedding])
            colors += ["green"] * len(all_actions)
            sizes += [10] * len(all_actions)
        else:
            all_action_embedding = action_embedding

        self.action_sc.set_offsets(all_action_embedding[:, :2])
        self.action_sc.set_color(colors)
        self.action_sc.set_sizes(sizes)

        self._figure_1.canvas.flush_events()
        self._figure_2.canvas.flush_events()
        self._figure_1.canvas.draw()
        self._figure_2.canvas.draw()

    def _report_result_upon_completion(self):
        pass

    def run_single_episode_as_kitchen(self):
        obs_history = []
        action_history = []
        latent_history = []
        o = np.random.choice(range(10))
        self.env.change_subtask(o)
        obs = self.env.reset()
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        option = torch.Tensor([[o]]).to(int).to('cuda')
        print('First optin chosen', option)
        self.curr_option = option
        last_obs = obs
        action, latents, (_, _) = self._get_action(obs, sample=True, keep_last_bins=False, option=option)
        done = False
        total_reward = 0
        obs_history.append(obs)
        action_history.append(action)
        latent_history.append(latents)

        n_queries = 0
        for i in range(self.cfg.num_eval_steps):
            # print(i)
            # try:
            if self.cfg.plot_interactions:
                self._plot_obs_and_actions(obs, action, done)
            if done:
                self._report_result_upon_completion()
                break
            if self.cfg.enable_render:
                self.env.render(mode="human")
            obs, reward, done, info = self.env.step(action)
            # obs = obs[:11]
            total_reward += reward
            if obs is None:
                obs = last_obs  # use cached observation in case of `None` observation
            else:
                last_obs = obs  # cache valid observation
            keep_last_bins = ((i + 1) % self.cfg.action_update_every) != 0
            action, latents, (option, option_logs) = self._get_action(
                obs, sample=True, keep_last_bins=keep_last_bins, option=option
            )
            print("Option:", option)
            if option != self.curr_option:
                self.env.change_subtask(option.item())
                self.curr_option = option

            obs_history.append(obs)
            action_history.append(action)
            latent_history.append(latents)
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        logging.info(f"Number of queries: {n_queries}")
        print("Number of queries", n_queries)
        return total_reward, obs_history, action_history, latent_history, info
    def get_obs(self, obs):
        if self.cfg.env.dataset_fn.use_image_data:
            agentview_img = obs['agentview_image']
            eye_in_hand_img = obs['robot0_eye_in_hand_image']
            with torch.no_grad():
                PREDICTOR.set_image(agentview_img)
            agentview_img = PREDICTOR.features.reshape(1,-1).cpu().numpy()
            agentview_img = pca_agentview.transform(agentview_img)

            with torch.no_grad():
                PREDICTOR.set_image(eye_in_hand_img)
            eye_in_hand_img = PREDICTOR.features.reshape(1,-1).cpu().numpy()
            eye_in_hand_img = pca_agentview.transform(eye_in_hand_img)

            obs = np.concatenate((
                obs['robot0_gripper_qpos'],
                obs['robot0_joint_pos'],
                agentview_img[0],
                eye_in_hand_img[0]
                ))
        else:    
            obs = np.concatenate((
                obs['robot0_gripper_qpos'],
                obs['robot0_joint_pos']))
        return obs
    
    def run_single_episode(self):

        from envs.libero.libero_goal import benchmark_instance, env_args, ControlEnv
        obs_history = []
        action_history = []
        latent_history = []
        task_names = [t.name for t in benchmark_instance.tasks]
        results = {}
        for o in range(10):
            results[o] = 0
            print("Testing task", task_names[o])
            bddl = benchmark_instance.get_task_bddl_file_path(o)
            
            env_args['bddl_file_name'] = bddl
            env = ControlEnv(**env_args)
            init_states = benchmark_instance.get_task_init_states(o)
            print(env.language_instruction)
            for init_state in init_states:
                env.reset()
                obs = env.set_init_state(init_state)
                obs = self.get_obs(obs)
                option = torch.Tensor([[o]]).to(int).to('cuda')
                last_obs = obs
                action, latents, (_, _) = self._get_action(
                    obs, sample=True, keep_last_bins=False, option=option)
                done = False
                total_reward = 0
                obs_history.append(obs)
                action_history.append(action)
                latent_history.append(latents)

                for i in range(self.cfg.num_eval_steps):
                    # print(i)
                    # try:
                    if self.cfg.plot_interactions:
                        self._plot_obs_and_actions(obs, action, done)
                    if done:
                        break
                    if self.cfg.enable_render:
                        env.env.render()
                    obs, reward, done, info = env.step(action)

                    obs = self.get_obs(obs)
                    # obs = env.get_sim_state()
                    # obs = obs[:11]
                    total_reward += reward
                    if obs is None:
                        obs = last_obs  # use cached observation in case of `None` observation
                    else:
                        last_obs = obs  # cache valid observation
                    keep_last_bins = ((i + 1) % self.cfg.action_update_every) != 0
                    action, latents, (option, option_logs) = self._get_action(
                        obs, sample=True, keep_last_bins=keep_last_bins, option=option
                    )
                    obs_history.append(obs)
                    action_history.append(action)
                    latent_history.append(latents)
                logging.info(f"Total reward: {total_reward}")
                results[o] += total_reward
                # logging.info(f"Final info: {info}")
                # logging.info(f"Number of queries: {n_queries}")
        # print("Number of queries", n_queries)
        logging.info(f"results{results}")
        return total_reward, obs_history, action_history, latent_history, info