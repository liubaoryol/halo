import logging
from collections import deque
from pathlib import Path

import einops
import gym
from gym.wrappers import RecordVideo
import hydra
import numpy as np
import torch
from models.bet.action_ae.generators.base import GeneratorDataParallel
from models.bet.latent_generators.latent_generator import LatentGeneratorDataParallel
import utils
import wandb


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        self.helper_procs = []

        self.env = gym.make(cfg.env.name)
        if cfg.record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=self.work_dir,
                episode_trigger=lambda x: x % 1 == 0,
            )

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None
        self.init_prob = None
        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()

        wandb.init(dir=self.work_dir, project=cfg.project, config=cfg._content)
        self.epoch = 0
        self.load_snapshot()

        # Set up history archival.
        self.window_size = cfg.window_size
        self.history = deque(maxlen=self.window_size)
        self.option_history = deque(maxlen=self.window_size)
        self.last_latents = None

        if self.cfg.flatten_obs:
            self.env = gym.wrappers.FlattenObservation(self.env)

        if self.cfg.plot_interactions:
            self._setup_plots()

        if self.cfg.start_from_seen:
            self._setup_starting_state()

    def _init_action_ae(self):
        if self.action_ae is None:  # possibly already initialized from snapshot
            self.action_ae = hydra.utils.instantiate(
                self.cfg.action_ae, _recursive_=False
            ).to(self.device)
            if self.cfg.data_parallel:
                self.action_ae = GeneratorDataParallel(self.action_ae)

    def _init_obs_encoding_net(self):
        if self.obs_encoding_net is None:  # possibly already initialized from snapshot
            self.obs_encoding_net = hydra.utils.instantiate(self.cfg.encoder)
            self.obs_encoding_net = self.obs_encoding_net.to(self.device)
            if self.cfg.data_parallel:
                self.obs_encoding_net = torch.nn.DataParallel(self.obs_encoding_net)

    def _init_state_prior(self):
        if self.state_prior is None:  # possibly already initialized from snapshot
            self.state_prior = hydra.utils.instantiate(
                self.cfg.state_prior,
                latent_dim=self.action_ae.latent_dim,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_plots(self):
        raise NotImplementedError

    def _setup_starting_state(self):
        raise NotImplementedError

    def _start_from_known(self):
        raise NotImplementedError

    def run_single_episode(self):
        sequence = [5, 6, 0, 1, 2, 3, 4]
        obs_history = []
        action_history = []
        latent_history = []
        obs = self.env.reset()
        # obs = obs[:11]
        # import pdb; pdb.set_trace()
        # initial option distribution has not been trained so will start with 0
        with utils.eval_mode(self.init_prob):
            option = self.init_prob(torch.Tensor(obs).to('cuda'))
            option_logs = torch.nn.Softmax()(option)
        print("Initial option distribution: ", option_logs)
        option = torch.multinomial(option_logs, num_samples=1).reshape(1,-1)
        o = sequence.pop(0)
        option = torch.Tensor([[o]]).to(int).to('cuda')
        self.curr_option = option
        last_obs = obs
        if self.cfg.start_from_seen:
            obs = self._start_from_known()
        action, latents, (_, _) = self._get_action(obs, sample=True, keep_last_bins=False, option=option)
        # option = torch.Tensor([[o]]).to(int).to('cuda')
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
            # print("Option:", option)
            if option != self.curr_option:
                if not len(sequence)==0:
                    # raise BufferError
                    name_option = self.env.ALL_TASKS[option.cpu().item()]
                    print("Algorithm selected ", name_option, "with probability", option_logs)
                    o = sequence.pop(0)
                    print("Executing task number", o)
                    option = torch.Tensor([[o]]).to(int).to('cuda')
                    self.curr_option = option
            obs_history.append(obs)
            action_history.append(action)
            latent_history.append(latents)
            # except KeyboardInterrupt:
            #     import pdb; pdb.set_trace()
            # except BufferError:
            #     print("BufferError")
            #     import pdb; pdb.set_trace()
        logging.info(f"Total reward: {total_reward}")
        logging.info(f"Final info: {info}")
        logging.info(f"Number of queries: {n_queries}")
        print("Number of queries", n_queries)
        return total_reward, obs_history, action_history, latent_history, info

    def _report_result_upon_completion(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        print(obs, chosen_action, done)
        raise NotImplementedError

    def _get_action(self, obs, sample=False, keep_last_bins=False, option=None):
        with utils.eval_mode(
            self.action_ae, self.obs_encoding_net, self.state_prior, no_grad=True
        ):
            obs = torch.from_numpy(obs).float().to(self.cfg.device).unsqueeze(0)
            enc_obs = self.obs_encoding_net(obs).squeeze(0)
            enc_obs = einops.repeat(
                enc_obs, "obs -> batch obs", batch=self.cfg.action_batch_size
            )
            # Now, add to history. This automatically handles the case where
            # the history is full.

            # import pdb; pdb.set_trace()
            self.history.append(enc_obs)
            self.option_history.append(option)
            if self.cfg.use_state_prior:
                enc_obs_seq = torch.stack(tuple(self.history), dim=0)  # type: ignore
                try:
                    option = torch.concatenate(tuple(self.option_history), dim=1)
                except:
                    import pdb; pdb.set_trace()
                # Sample latents from the prior
                latents, (option, option_logs) = self.state_prior.generate_latents(
                    enc_obs_seq,
                    torch.ones_like(enc_obs_seq).mean(dim=-1),
                    option=option
                )
                # For visualization, also get raw logits and offsets
                # placeholder_target = (
                #     torch.zeros_like(latents[0]),
                #     torch.zeros_like(latents[1]),
                # )
                # (
                #     logits_to_save,
                #     offsets_to_save,
                # ), _ = self.state_prior.get_latent_and_loss(enc_obs_seq, placeholder_target)
                logits_to_save, offsets_to_save = None, None

                offsets = None
                if type(latents) is tuple:
                    latents, offsets = latents

                if keep_last_bins and (self.last_latents is not None):
                    latents = self.last_latents
                else:
                    self.last_latents = latents

                # Take the final action latent
                if self.cfg.enable_offsets:
                    action_latents = (latents[:, -1:, :], offsets[:, -1:, :])
                else:
                    action_latents = latents[:, -1:, :]
            else:
                action_latents = self.action_ae.sample_latents(
                    num_latents=self.cfg.action_batch_size
                )
            actions = self.action_ae.decode_actions(
                latent_action_batch=action_latents,
                input_rep_batch=enc_obs,
            )
            actions = actions.cpu().numpy()
            if sample:
                sampled_action = np.random.randint(len(actions))
                actions = actions[sampled_action]
                # (seq==1, action_dim), since batch dim reduced by sampling
                actions = einops.rearrange(actions, "1 action_dim -> action_dim")
            else:
                # (batch, seq==1, action_dim)
                actions = einops.rearrange(
                    actions, "batch 1 action_dim -> batch action_dim"
                )
            return actions, (logits_to_save, offsets_to_save, action_latents), (option, option_logs)

    def run(self):
        rewards = []
        infos = []
        if self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        for i in range(self.cfg.num_eval_eps):
            reward, obses, actions, latents, info = self.run_single_episode()
            rewards.append(reward)
            infos.append(info)
            torch.save(actions, Path.cwd() / f"actions_{i}.pth")
            torch.save(latents, Path.cwd() / f"latents_{i}.pth")
        self.env.close()
        logging.info(rewards)
        logging.info(infos)
        return rewards, infos

    @property
    def snapshot(self):
        return Path(self.cfg.load_dir or self.work_dir) / "snapshot.pt"

    def load_snapshot(self):
        keys_to_load = ["action_ae", "obs_encoding_net", "state_prior", "init_prob"]
        with self.snapshot.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        loaded_keys = []
        for k, v in payload.items():
            if k in keys_to_load:
                loaded_keys.append(k)
                self.__dict__[k] = v.to(self.cfg.device)

        if len(loaded_keys) != len(keys_to_load):
            raise ValueError(
                "Snapshot does not contain the following keys: "
                f"{set(keys_to_load) - set(loaded_keys)}"
            )
