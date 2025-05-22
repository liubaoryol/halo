import torch
import wandb
import os
from datetime import datetime
from typing import Union, Tuple, Dict, Optional
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

from dataloaders.utils_hbc import augmentTrajectoryWithLatent, TrajectoryWithLatent

timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

from stable_baselines3.common.evaluation import evaluate_policy


class Evaluator():
    def __init__(self, logger, env):
        self._logger = logger
        self._env = env
        
    def evaluate_and_log(
        self,
        model, 
        student,
        oracle,
        epoch
        ):

        hamming_train = self.hamming_distance(student.demos, oracle.true_options)
        hamming_test = self.hamming_distance(student.demos_test, oracle.true_options_test)

        self._logger.log_batch(
            epoch_num=epoch,
            hamming_loss=hamming_train,
            hamming_loss_test=hamming_test
        )

        self._logger.last_01_distance = hamming_train
    
    def hamming_distance(self, demos, true_options):
        with torch.no_grad():
            options = [demo.latent for demo in demos]
            f = lambda x: np.linalg.norm(
                (options[x].squeeze()[1:] - true_options[x]), 0)/len(options[x])
            distances = list(map(f, range(len(options))))
        return np.mean(distances)
    
    def env_interaction(self, model):
        mean_return, std_return = evaluate_policy(model, self._env, 10)
        return mean_return, std_return

class HBCLoggerPartial(bc.BCLogger):
    def __init__(self, logger, lo: bool, wandb_run=None):
        super().__init__(logger)
        self.part = 'lo' if lo else 'hi'
        self.wandb_run = wandb_run

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics,
        rollout_stats,
    ):
        for k, v in training_metrics.__dict__.items():
            name = f"bc_{self.part}/{k}"
            value = float(v) if v is not None else None
            self._logger.record(name, value)
            if self.wandb_run is not None:
                self.wandb_run.log({name: value})

class HBCLogger:
    """Utility class to help logging information relevant to Behavior Cloning."""

    def __init__(self, logger: imit_logger.HierarchicalLogger, wandb_run=None,
                 student_type: str=None):
        """Create new BC logger.

        Args:
            logger: The logger to feed all the information to.
        """
        self.student_type = student_type
        self._logger = logger
        self._tensorboard_step = 0
        self._current_epoch = 0
        self._logger_lo = HBCLoggerPartial(logger,
                                           lo=True,
                                           wandb_run=wandb_run)
        self._logger_hi = HBCLoggerPartial(logger,
                                           lo=False,
                                           wandb_run=wandb_run)
        self.wandb_run = wandb_run
        self.metrics_table = wandb.Table(
            columns=[
                'model',
                'student_type',
                'epoch',
                'hamming_train',
                'hamming_test'
            ])

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_batch(
        self,
        epoch_num: int,
        hamming_loss: int,
        hamming_loss_test: int
    ):
        self._logger.record("env/epoch", epoch_num)
        self._logger.record("env/hamming_loss_train", hamming_loss)
        self._logger.record("env/hamming_loss_test", hamming_loss_test)

        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

        self.metrics_table.add_data(
            'hbc',
            self.student_type,
            epoch_num,
            hamming_loss,
            hamming_loss_test
            )

        if self.wandb_run is not None:
            self.wandb_run.log({
                "env/epoch": epoch_num,
                "env/hamming_loss": hamming_loss,
                "env/hamming_loss_test": hamming_loss_test
            }, step=epoch_num)


    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state


class HBC:
    def __init__(self,
                 option_dim: int,
                 device: str,
                 env,
                 work_dir,
                 exp_identifier='hbc',
                 curious_student=None,
                 results_dir='results',
                 wandb_run=None
                 ):
        self.device = device
        self.option_dim = option_dim
        self.curious_student = curious_student
        self.env = env
        self.work_dir = work_dir

        env_id = f'size{env.size}-targets{env.n_targets}'
        logging_dir = os.path.join(
            work_dir,
            f'{results_dir}/{env_id}/{exp_identifier}_{timestamp()}/'
            )
        self.env_id = env_id
        new_logger = imit_logger.configure(logging_dir, ["stdout"])
        student_type = self.curious_student.student_type
        self._logger = HBCLogger(new_logger, wandb_run, student_type)


        obs_space = env.observation_space
        new_lo = np.concatenate([obs_space.low, [0]])
        new_hi = np.concatenate([obs_space.high, [option_dim]])
        rng = np.random.default_rng(0)

        self.policy_lo = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=env.action_space, # Check as sometimes it's continuosu
            rng=rng,
            device=device
        )
        self.policy_lo._bc_logger = self._logger._logger_lo
        new_lo[-1] = -1
        self.policy_hi = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=spaces.Discrete(option_dim),
            rng=rng,
            device=device
        )
        self.policy_hi._bc_logger = self._logger._logger_hi
        TrajectoryWithLatent.set_policy(self.policy_lo, self.policy_hi)

        self.evaluator = Evaluator(self._logger, self.env)
        

    def train(self, n_epochs=1):
        transitions_lo, transitions_hi = self.transitions(self.curious_student.demos) 
        self.policy_lo.set_demonstrations(transitions_lo)
        self.policy_lo.train(n_epochs=n_epochs)
        self.policy_hi.set_demonstrations(transitions_hi)
        self.policy_hi.train(n_epochs=n_epochs)


    def transitions(self, expert_demos):
        expert_lo = []
        expert_hi = []
        for demo in expert_demos:
            opts = demo.latent
            expert_lo.append(imitation.data.types.TrajectoryWithRew(
                obs = np.concatenate([demo.obs, opts[1:]], axis=1),
                acts = demo.acts,
                infos =  demo.infos,
                terminal = demo.terminal,
                rews = demo.rews
            )
            )
            expert_hi.append(
                imitation.data.types.TrajectoryWithRew(
                    obs = np.concatenate([demo.obs, opts[:-1]], axis=1),
                    acts = opts[1:-1].reshape(-1),
                    infos = demo.infos,
                    terminal = demo.terminal,
                    rews = demo.rews
                ))
        transitions_hi = rollout.flatten_trajectories(expert_hi)
        transitions_lo = rollout.flatten_trajectories(expert_lo)

        return transitions_lo, transitions_hi

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        if state is None:
            n = len(observation)
            state = -np.ones(n)
        state[episode_start] = -1

        if observation.dtype==object:
            observation = observation.astype(float)

        hi_input = obs_as_tensor(
            np.concatenate([observation, state.reshape(-1, 1)], axis=1),
            device=self.device
            )
        state, _ = self.policy_hi.policy.predict(hi_input)
        lo_input = obs_as_tensor(
            np.concatenate([observation, state.reshape(-1, 1)], axis=1),
            device=self.device
            )
        actions, _ = self.policy_lo.policy.predict(lo_input)
        return actions, state

    def save(self, ckpt_num: int=1):
        base_path = os.path.join(self.work_dir, 'checkpoints')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        path = os.path.join(base_path, f'checkpoint-epoch-{ckpt_num}.tar')
        policy_lo_state = self.policy_lo.policy.state_dict()
        policy_hi_state = self.policy_hi.policy.state_dict()
        torch.save({
            'policy_lo_state': policy_lo_state,
            'policy_hi_state': policy_hi_state
            }, path)
        if self._logger.wandb_run is not None:
            wandb.save(path, base_path=self.work_dir)

    def load(self, path):
        model_state_dict = torch.load(path)
        self.policy_lo.policy.load_state_dict(model_state_dict['policy_lo_state'])
        self.policy_hi.policy.load_state_dict(model_state_dict['policy_hi_state'])
