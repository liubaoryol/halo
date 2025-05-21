from typing import List
import dataclasses
import torch
import numpy as np
from scipy.stats import entropy
from scipy.special import logsumexp

from stable_baselines3.common.utils import obs_as_tensor
from imitation.data.types import TrajectoryWithRew

def augmentTrajectoryWithLatent(trajectories: List[TrajectoryWithRew], option_dim):
    """Get list with augmented trajectories with latent variables"""

    opts_trajs = []
    for traj in trajectories:
        tmp = TrajectoryWithLatent(
            obs=traj.obs,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal,
            rews=traj.rews
        )
        opts_trajs.append(tmp)

    TrajectoryWithLatent.set_option_dim(option_dim)
    return opts_trajs

@dataclasses.dataclass(frozen=True)
class TrajectoryWithLatent(TrajectoryWithRew):
    """A batch of obs-act-lat-is_estimated-info-done transitions.
    
    This class has a 'global policy' that will be used for estimating
    latent states with Viterbi algorithm. This policy will be updated 
    automatically as hbc is updated, as it's pointing to the same obj.
    """

    _option_dim: int = None
    _latent: np.ndarray = None
    _is_latent_estimated: np.ndarray = None
    """
    Reward. Shape: (batch_size, ). dtype float.

    The reward `rew[i]` at the i'th timestep is received after the
    agent has taken action `acts[i]`.
    """

    @classmethod
    def set_policy(cls, policy_lo, policy_hi):
        cls._policy_lo = policy_lo.policy
        cls._policy_hi = policy_hi.policy
        cls._device = cls._policy_lo.device
    
    @classmethod
    def set_option_dim(cls, option_dim):
        cls.option_dim = option_dim

    def __post_init__(self):
        """Performs input validation, including for rews."""
        super().__post_init__()
        object.__setattr__(self,
                           '_is_latent_estimated',
                           np.concatenate(
                               [[True], np.zeros(len(self.obs), dtype=bool)]
                               ))
        object.__setattr__(self, '_latent', -np.ones(len(self.obs) + 1, dtype=int))
        object.__setattr__(self, '_N', len(self.obs))
        object.__setattr__(self, '_device', 'cpu')
    
    @property
    def latent(self):
        if not getattr(self, '_policy_lo', None):
            raise AttributeError('Policies are not defined. Set low and high class policies')
        return self._update_latent()

    def set_true_latent(self, idx, value):
        """Set true latent, returns boolean variable indicating completion
        
        Returns True if setting was possible, False if it was set up already."""
        if not self._is_latent_estimated[idx+1]:
            self._latent[idx + 1] = value
            self._is_latent_estimated[idx + 1] = True
            return True
        return False
    
    def _log_prob_action(self):
        """
        Return probability P(a|s, o) N x option_dim
        """
        states = obs_as_tensor(self.obs[:-1], self._device)
        actions = obs_as_tensor(self.acts, self._device)
        results = []
        for o in range(self.option_dim):
            input_o = torch.concat([states, torch.ones([self._N-1,1])*o], axis=1)
            log_prob=self._policy_lo.get_distribution(input_o).log_prob(actions)
            results.append(log_prob)
        return torch.stack(results, axis=1)
    
    def _log_prob_option(self):
        """
        Return probability P(o|s, o_); size N x (option_dim + 1) x option_dim
        """
        states = obs_as_tensor(self.obs, self._device)
        results = []
        for o in range(-1, self.option_dim):
            input_o = torch.concat([states, torch.ones([self._N,1])*o], axis=1)
            log_prob = self._policy_hi.get_distribution(input_o).distribution.logits.detach()
            results.append(log_prob)
        return torch.stack(results, axis=1)
    
    def _update_latent(self):
        """Apply Viterbi algorithm"""
        with torch.no_grad():
            log_acts = self._log_prob_action()  # demo_len x 1 x ct
            log_opts = self._log_prob_option()  # demo_len x (ct_1+1) x ct
            # Special handling of last state:
            log_acts = log_acts.reshape([-1, 1, self.option_dim])
            last_log_opts = log_opts[-1][1:]
            log_opts = log_opts[:-1]

            # Done special handling
            log_prob = log_opts[:, 1:] + log_acts
            log_prob = torch.concatenate([log_prob, last_log_opts.unsqueeze(0)])

            # forward
            max_path = torch.empty(self._N,
                                   self.option_dim,
                                   dtype=torch.long,
                                   device=self._device)
            accumulate_logp = torch.zeros(self.option_dim) 
            for i in range(self._N):
                if self._is_latent_estimated[i]:
                    accumulate_logp, max_path[i, :] = accumulate_logp + torch.zeros([self.option_dim]), self._latent[i] * torch.ones([self.option_dim])
                else:
                    accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = -torch.ones(self._N+1, 1, dtype=torch.long, device=self._device)
            log_prob_traj, idx = accumulate_logp.max(dim=-1)
            c_array[-1] = max_path[-1][idx]
            for i in range(self._N, 1, -1):
                c_array[i-1] = max_path[i-1][c_array[i]]
        return c_array.detach().numpy()

    def forward(self):
        """Forward messages"""
        with torch.no_grad():
            log_acts = self._log_prob_action()  # demo_len x 1 x ct
            log_opts = self._log_prob_option()  # demo_len x (ct_1+1) x ct
            # Special handling of last state:
            log_acts = log_acts.reshape([-1, 1, self.option_dim])
            last_log_opts = log_opts[-1][1:]
            log_opts = log_opts[:-1]

            # Done special handling
            log_prob = log_opts[:, 1:] + log_acts
            log_prob = torch.concatenate([log_prob, last_log_opts.unsqueeze(0)])

        log_prob = log_prob.detach().numpy()
        forward_msg = np.zeros([self._N+1, 1, self.option_dim])

        for j in range(1, self._N+1):
            if self._is_latent_estimated[j]:    
                forward_msg[j] = -np.inf
                forward_msg[j][0][self._latent[j]] = 0
            # if self._is_latent_estimated[j]:
            #     forward_msg[j] = np.log(1e-300)
            #     forward_msg[j][:, self._latent[j]] = 0
            else:
                forward_msg[j] = logsumexp(np.concatenate(
                    [forward_msg[j-1]]*self.option_dim) + log_prob[j-1], axis=1) # or 1?
        return forward_msg

    def backward(self):
        """Backward messages"""
        
        with torch.no_grad():
            log_acts = self._log_prob_action()  # demo_len x 1 x ct
            log_opts = self._log_prob_option()  # demo_len x (ct_1+1) x ct
            # Special handling of last state:
            log_acts = log_acts.reshape([-1, 1, self.option_dim])
            log_opts = log_opts[1:]

            # Done special handling
            log_prob = log_opts[:, 1:] + log_acts
        
        log_prob = log_prob.detach().numpy()
        backward_msg = np.zeros([self._N+1, 1, self.option_dim])
        for j in reversed(range(1, self._N)):
            backward_msg[j] = logsumexp(np.concatenate(
                [backward_msg[j-1]]*self.option_dim) + log_prob[j-1], axis=1)
        return backward_msg

    def entropy(self):
        res = self.forward() + self.backward()
        normalizer = logsumexp(res, axis=-1)
        normalizer = np.stack([normalizer]*self.option_dim, axis=-1)
        res -= normalizer
        return entropy(np.exp(res), axis=(1,2))

    def __eq__(self, other:TrajectoryWithRew):
        return (other.obs == self.obs and  
                other.acts == self.acts and
                other.infos == self.infos and
                other.rews == self.rews and
                other.terminal == self.terminal)