import sys
import os
sys.path.append(os.getcwd())
import wandb
import gymnasium as gym
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
import torch
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import tqdm

import students.hbc_students as students

from envs.rescueworld.rw import FilterLatent
from models.hbc.hbc import HBC
import envs
import utils
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')



class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("Saving to {}".format(self.work_dir))
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_options = cfg.env.latent_dim

        utils.set_seed_everywhere(cfg.seed)

        self.env = gym.make(cfg.env.name, **{
            'size': cfg.env.size,
            'n_targets': cfg.env.n_targets,
            'allow_variable_horizon': cfg.env.allow_variable_horizon,
            'fixed_targets': cfg.env.fixed_targets,
            'danger': cfg.env.danger,
            'obstacles': cfg.env.obstacles
            })
        self.env = Monitor(self.env)
        self.env = FilterLatent(
            self.env,
            list(range(cfg.env.filter_state_until, 0))
        )
        horizon = cfg.env.size**2 * cfg.env.n_targets
        self.env.unwrapped._max_episode_steps = horizon

        self.oracle = hydra.utils.call(cfg.env.dataset_fn)
        # Create student
        self.student = getattr(students, cfg.student_type.capitalize())(
            option_dim=self.num_options,
            oracle=self.oracle) #TODO
        if cfg.student_type=='random':
            self.student.query_percent=cfg.randomst_query_percent
            self.student.student_type=f'query_percent{cfg.randomst_query_percent}'

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

        # Create the model
        self.final_hbc = HBC(
            option_dim=cfg.env.n_targets,
            device=cfg.device,
            env=self.env,
            work_dir=self.work_dir,
            exp_identifier=self.wandb_run.name,
            curious_student=self.student,
            wandb_run=self.wandb_run
            )
        self.hbc = HBC(
            option_dim=cfg.env.n_targets,
            device=cfg.device,
            env=self.env,
            work_dir=self.work_dir,
            exp_identifier=self.wandb_run.name,
            curious_student=self.student,
            wandb_run=self.wandb_run
            )

        if self.student.single_query_only:
            print("Querying oracle!")
            self.student.query_oracle()
        # Print data stats for query budget
        num_sa_pairs = [len(data.obs) for data in self.oracle.expert_trajectories]
        num_sa_pairs = sum(num_sa_pairs)
        self.query_budget = cfg.query_percentage_budget * num_sa_pairs
        print('NUMBER OF QUERIES: \n',
              cfg.query_percentage_budget*100, '%',
              ' percent of queries, equivalent to ',
              self.query_budget, 'number of queries')
        self.num_queries = cfg.num_queries
        self.query_freq = cfg.query_freq


    def run(self):
        self.query_time = 0
        self.iterator = tqdm.trange(0, self.cfg.num_epochs)
        self.iterator.set_description("Training Hierarchical BC: ")
        for epoch in self.iterator:
            self.query_time +=1
            if not self.student.single_query_only:
                budget_available = self.student._num_queries < self.query_budget
                if not self.query_time%self.query_freq:
                    if budget_available:
                        self.student.query_oracle(num_queries=self.cfg.num_queries)
            
            self.hbc.train()
            if ((epoch + 1) % self.cfg.eval_every) == 0:
                self.hbc.evaluator.evaluate_and_log(
                    model=self.hbc,
                    student=self.student,
                    oracle=self.student.oracle,
                    epoch=epoch)

            if not epoch % self.cfg.save_every:
                self.hbc.save(ckpt_num=epoch)
    
        import numpy as np
        artifact = wandb.Artifact("final_summary", type="summary")
        text_file = os.path.join(self.work_dir, "final_summary.txt")

        self.final_hbc.train(self.cfg.num_policy_only_epochs)
        mean_return, std_return = self.hbc.evaluator.env_interaction(self.hbc)
        summary_string1 = f"{mean_return:.2f} +/- {std_return:.2f}\n"
        mean_return, std_return = self.hbc.evaluator.env_interaction(self.final_hbc)
        summary_string2 = f"Retrained hbc: {mean_return:.2f} +/- {std_return:.2f}\n"
        with open(text_file, "w") as f:
            f.write(summary_string1)
            f.write(summary_string2)

        artifact.add_file(text_file)
        self.wandb_run.log_artifact(artifact)
        # Final log: table with metrics
        self.wandb_run.log({"metrics/metrics": self.hbc._logger.metrics_table})
        import time
        time.sleep(2)

@hydra.main(config_path="configs", config_name="train_rw_n2")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()