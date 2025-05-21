import logging
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import numpy as np
import tqdm
import hydra

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from models.bet.action_ae.generators.base import GeneratorDataParallel
from models.bet.latent_generators.latent_generator import LatentGeneratorDataParallel
import students
import utils
import wandb


timestamp = lambda: datetime.now().strftime('-%m-%d_%H-%M')

class InitialStateData(Dataset):
    def __init__(self, dataset, option_dim):
        self.dataset = dataset
        self.option_dim = option_dim
        super().__init__()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return (item[0][0],
                item[1][0],
                item[2][0],
                torch.nn.functional.one_hot(item[3][0].to(torch.long), num_classes = self.option_dim)
        )
    def __len__(self):
        return int(self.dataset.masks.size(0))


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_options = cfg.env.latent_dim

        utils.set_seed_everywhere(cfg.seed)
        # TODO: adjusto to include libero

        # self._setup_loaders()

        # Create the model
        self.action_ae = None
        self.obs_encoding_net = None
        self.state_prior = None

        if not self.cfg.lazy_init_models:
            self._init_action_ae()
            self._init_obs_encoding_net()
            self._init_state_prior()
        self.dataset = hydra.utils.call(
            cfg.env.dataset_fn,
            train_fraction=cfg.train_fraction,
            random_seed=cfg.seed,
            device=self.device
        )
        self.train_set, self.test_set = self.dataset
        self._setup_loaders()
        self.init_data = InitialStateData(
            dataset=self.train_set.dataset.dataset,
            option_dim=self.num_options)
        # self.init_data = minorCustomData(data_directory=cfg.env.dataset_fn.data_directory, device=cfg.device, option_dim=self.num_options)
        self.init_dataloader = DataLoader(self.init_data, batch_size=64, shuffle=True)

        # Simple MLP to sample an initial option.
        self.init_prob = torch.nn.Sequential(
            torch.nn.Linear(self.init_data[0][0].size(0), self.init_data[0][0].size(0)//2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.init_data[0][0].size(0)//2, self.num_options),
        ).to(self.device)
        self.init_optimizer = torch.optim.Adam(
            self.init_prob.parameters()
            )
        self.init_criterion = torch.nn.CrossEntropyLoss()

        self.student = getattr(students, cfg.student_type.capitalize())(
            option_dim=self.num_options,
            state_prior=self.state_prior,
            action_ae=self.action_ae,
            dataset=self.train_set.dataset.dataset)
        if cfg.student_type=='random':
            self.student.query_percent=cfg.randomst_query_percent
            self.student.student_type=f'query_percent{cfg.randomst_query_percent}'
        num_sa_pairs = self.train_set.dataset.dataset.masks.sum()
        self.query_budget = cfg.query_percentage_budget * num_sa_pairs
        print('NUMBER OF QUERIES: \n',
              cfg.query_percentage_budget*100, '%',
              ' percent of queries, equivalent to ',
              self.query_budget, 'number of queries')
        self.num_queries = cfg.num_queries
        self.query_freq = cfg.query_freq

        self.log_components = OrderedDict()
        self.epoch = self.prior_epoch = self.option_epoch = 0

        self.save_training_latents = False
        self._training_latents = []

        self.wandb_run = wandb.init(
            dir=str(self.work_dir),
            project=cfg.project,
            name=self.student.student_type+'Student'+timestamp(),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update(
            {
                "save_path": self.work_dir,
            }
        )

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
                latent_dim=self.num_options,
                vocab_size=self.action_ae.num_latents,
            ).to(self.device)
            if self.cfg.data_parallel:
                self.state_prior = LatentGeneratorDataParallel(self.state_prior)
            self.state_prior_optimizer = self.state_prior.get_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )
            self.option_optimizer = self.state_prior.get_option_optimizer(
                learning_rate=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=tuple(self.cfg.betas),
            )

    def _setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.latent_collection_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


    def train_prior(self):
        self.query_time = 0
        self.state_prior.train()
        with utils.eval_mode(self.obs_encoding_net, self.action_ae):
            pbar = tqdm.tqdm(
                self.train_loader, desc=f"Training prior epoch {self.prior_epoch}"
            )
            for data in pbar:
                self.query_time +=1
                if not self.student.single_query_only:
                    budget_available = self.student._num_queries < self.query_budget
                    if not self.query_time%self.query_freq:
                        
                        trjs_changed = set(np.random.randint(
                            len(self.init_data), size=self.num_queries))
                        if budget_available:
                            trjs_changed = self.student.query_oracle(
                                self.train_set.dataset.dataset.oracle,
                                num_queries=self.num_queries)
                        for traj_num in trjs_changed:
                            self.train_set.dataset.dataset.update_options_of_traj(
                                self.student,
                                traj_num)
                if self.cfg.env.dataset_fn.use_image_data:
                    observations, action, mask, option, gt_option, idx = data
                else:
                    observations, action, mask, option, gt_option = data
                self.state_prior_optimizer.zero_grad(set_to_none=True)
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)

                if self.cfg.env.dataset_fn.use_image_data:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                        idx= idx,
                        dataset = self.train_set.dataset.dataset
                    )
                else:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                    )
                    
                logits, loss2 = self.state_prior.option_model((enc_obs[:, :-1], option[:, :-1]), option[:,1:])
                loss.backward()
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.state_prior.parameters(), self.cfg.grad_norm_clip
                )
                self.state_prior_optimizer.step()
                self.option_optimizer.step()

                # Book keeping
                targets = gt_option.to(enc_obs.device)[:,1:]
                targets = F.one_hot(targets.to(torch.int64), num_classes=self.num_options).to(torch.float)
                loss3 = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1, logits.size(-1)))
                loss4 = (gt_option!=option).sum() / option.numel()
                self.log_append("option_train", len(observations), {
                    'cross_entropy': loss2, 
                    'gt_cross_entropy': loss3,
                    'fwbw_estimation': loss4})
                self.log_append("prior_train", len(observations), loss_components)
    
    def train_action_policy_from_scratch(self):
        # Assume labeling is done, so querying is not part of this anymore

        self.state_prior.train()
        
        with utils.eval_mode(self.obs_encoding_net, self.action_ae):
            pbar = tqdm.tqdm(
                self.train_loader, desc=f"Training policy epoch {self.prior_epoch}"
            )
            for data in pbar:

                if self.cfg.env.dataset_fn.use_image_data:
                    observations, action, mask, option, gt_option, idx = data
                else:
                    observations, action, mask, option, gt_option = data
                self.state_prior_optimizer.zero_grad(set_to_none=True)
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                if self.cfg.env.dataset_fn.use_image_data:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                        idx= idx,
                        dataset = self.train_set.dataset.dataset
                    )
                else:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.state_prior.parameters(), self.cfg.grad_norm_clip
                )
                self.state_prior_optimizer.step()

                # Book keeping
                targets = gt_option.to(enc_obs.device)[:,1:]
                targets = F.one_hot(targets.to(torch.int64), num_classes=self.num_options).to(torch.float)

                self.log_append("final_policy_train", len(observations), loss_components)

    def eval_prior(self, id=''):
        with utils.eval_mode(
            self.obs_encoding_net, self.action_ae, self.state_prior, no_grad=True
        ):
            for data in self.test_loader:
                if self.cfg.env.dataset_fn.use_image_data:
                    observations, action, mask, option, gt_option, idx = data
                else:
                    observations, action, mask, option, gt_option = data
                obs, act = observations.to(self.device), action.to(self.device)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                if self.cfg.env.dataset_fn.use_image_data:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                        idx= idx,
                        dataset = self.train_set.dataset.dataset
                    )
                else:
                    _, loss, loss_components = self.state_prior.get_latent_and_loss(
                        obs_rep=(enc_obs, option),
                        target_latents=latent,
                        return_loss_components=True,
                    )
                self.log_append(id + "prior_eval", len(observations), loss_components)

                _, loss2 = self.state_prior.option_model((enc_obs[:, :-1], option[:, :-1]), option[:, 1:])
                
                targets = gt_option.to(enc_obs.device)[:,1:]
                targets = F.one_hot(targets.to(torch.int64), num_classes=self.num_options).to(torch.float)
                loss3 = F.cross_entropy(_.view(-1, _.size(-1)), targets.view(-1, _.size(-1)))
                loss4 = (gt_option!=option).sum() / option.numel()
                self.log_append(id + "option_eval", len(observations), {
                    'cross_entropy': loss2, 
                    'gt_cross_entropy': loss3,
                    'fwbw_estimation': loss4})

    def train_init_state(self, epoch):
        # Train for one epoch
        self.init_prob.train()
        total = 0
        for observations, _, _, option in self.init_dataloader:
            self.init_optimizer.zero_grad()
            obs, targets = observations.to(self.device), option.to(self.device)
            logits = self.init_prob(obs)
            loss = self.init_criterion(logits, targets.to(torch.float))
            total += loss.item()
            loss.backward()
            self.init_optimizer.step()
        print("Training init distr; epoch ", epoch, "with loss", total)
        
    def run(self):
        snapshot = self.snapshot
        if snapshot.exists():
            print(f"Resuming: {snapshot}")
            self.load_snapshot()

        if self.cfg.lazy_init_models:
            self._init_obs_encoding_net()
            self._init_action_ae()
        self.action_ae.fit_model(
            self.train_loader,
            self.test_loader,
            self.obs_encoding_net,
        )
        
        # Train the action prior and option model.
        if self.cfg.lazy_init_models:
            self._init_state_prior()
        self.log_components = OrderedDict()

        if self.student.single_query_only:
            print("Querying oracle!")
            self.train_set.dataset.dataset.query_oracle(self.student)

        
        self.state_prior_iterator = tqdm.trange(
            self.prior_epoch, self.cfg.num_prior_epochs
        )
        self.state_prior_iterator.set_description("Training prior: ")

        for epoch in self.state_prior_iterator:
            self.prior_epoch = epoch
            # update options every epoch
            self.train_set.dataset.dataset.update_options(self.student) 
            self.train_init_state(epoch)
            self.train_prior()  # Trains prior and queries oracle
            if ((self.prior_epoch + 1) % self.cfg.eval_prior_every) == 0:
                self.eval_prior()
            self.flush_log(epoch=epoch + self.epoch, iterator=self.state_prior_iterator)
            self.prior_epoch += 1
            if ((self.prior_epoch + 1) % self.cfg.save_prior_every) == 0:
                self.save_snapshot()

        tag_func = (
            lambda m: m.module.__class__.__name__
            if self.cfg.data_parallel
            else m.__class__.__name__
        )
        tags = tuple(
            map(tag_func, [self.obs_encoding_net, self.action_ae, self.state_prior])
        )
        self.wandb_run.tags += tags

        self.state_prior.model.apply(self.state_prior.model._init_weights)
        self.state_policy_only_iterator = tqdm.trange(
            0, self.cfg.num_policy_only_epochs
        )
        self.state_policy_only_iterator.set_description("Training policy only: ")
        self.epoch_continue = epoch + self.epoch
        for epoch in self.state_policy_only_iterator:
            self.prior_epoch = epoch
            self.train_init_state(epoch)
            self.train_action_policy_from_scratch()  # Trains prior and queries oracle
            if ((self.prior_epoch + 1) % self.cfg.eval_prior_every) == 0:
                self.eval_prior(id='post_option_')
            self.flush_log(epoch=epoch + self.epoch_continue, iterator=self.state_policy_only_iterator)
            self.prior_epoch += 1
            if ((self.prior_epoch + 1) % self.cfg.save_prior_every) == 0:
                self.save_snapshot()

        tag_func = (
            lambda m: m.module.__class__.__name__
            if self.cfg.data_parallel
            else m.__class__.__name__
        )
        tags = tuple(
            map(tag_func, [self.obs_encoding_net, self.action_ae, self.state_prior])
        )
        self.wandb_run.tags += tags

    @property
    def snapshot(self):
        return self.work_dir / "snapshot.pt"

    def save_snapshot(self):
        self._keys_to_save = [
            "action_ae",
            "obs_encoding_net",
            "epoch",
            "prior_epoch",
            "state_prior",
            "init_prob"
        ]
        payload = {k: self.__dict__[k] for k in self._keys_to_save}
        with self.snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        with self.snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        not_in_payload = set(self._keys_to_save) - set(payload.keys())
        if len(not_in_payload):
            logging.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
