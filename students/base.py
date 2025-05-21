"""Base classes for student and oracle"""

import os
import pickle
import logging
import dataclasses
import numpy as np
from typing import List, Any
from abc import ABC, abstractmethod

from models.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from models.bet.latent_generators.mingpt import MinGPT

@dataclasses.dataclass(frozen=True)
class QueryIdentifier:
    traj_num: int
    idx_query: int
    # previous_estimated: int
    # gt_latent_set: int

@dataclasses.dataclass
class Oracle:
    expert_trajectories: List[List] = None
    true_options: List[np.ndarray] = None
    expert_trajectories_test: List[List] = None
    true_options_test: List[np.ndarray] = None

    def query(self, trajectory_num, position_num):
        return self.true_options[trajectory_num][position_num]
    
    def __str__(self):
        return f'Oracle(num_demos={len(self.expert_trajectories)})'
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for name, attribute in self.__dict__.items():
            name += '.pkl'
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @property
    def max_queries(self) -> int:
        num_queries = 0
        for demo in self.expert_trajectories:
            num_queries+=len(demo.obs)
        return num_queries

    @classmethod
    def load(cls, path):
        my_model = {}
        for name in cls.__annotations__:
            
            with open("/".join((path, name+'.pkl')), "rb") as f:
                my_model[name] = pickle.load(f)
        return cls(**my_model)

    def __eq__(self, other):        
        A = other.expert_trajectories == self.expert_trajectories
        B = other.expert_trajectories_test == self.expert_trajectories_test
        C = np.all([(opts == other_opts).all() for opts, other_opts
                    in zip(self.true_options, other.true_options)])
        if self.true_options_test is None:
            D = other.true_options_test is None
        else:
            D = np.all([(opts == other_opts).all() for opts, other_opts 
                        in zip(self.true_options_test, other.true_options_test)])
        return A and B and C and D

    def stats(self):
        obs_len = [len(tr.obs) for tr in self.expert_trajectories]
        returns = [np.sum(tr.rews) for tr in self.expert_trajectories]
        obs_t_len = [len(tr.obs) for tr in self.expert_trajectories_test]
        returns_t = [np.sum(tr.rews) for tr in self.expert_trajectories_test]

        print("STATS TRAINING DATASET")
        print("Num trajectories: ", len(self.expert_trajectories))
        print("Avereage length: {}+/-{}".format(
            np.mean(obs_len),
            np.std(obs_len)))
        print("Average returns: {}+/-{}\n".format(
            np.mean(returns),
            np.std(returns)
        ))

        print("STATS TESTING DATASET")
        print("Num trajectories: ", len(self.expert_trajectories))
        print("Avereage length: {}+/-{}".format(
            np.mean(obs_t_len),
            np.std(obs_t_len)))
        print("Average returns: {}+/-{}".format(
            np.mean(returns_t),
            np.std(returns_t)
        ))

        return (np.mean(returns), np.std(returns)), (np.mean(returns_t), np.std(returns_t))


@dataclasses.dataclass
class CuriousPupil(ABC):
    option_dim: int
    annotated_options: np.ndarray = None
    state_prior: MinGPT = None
    action_ae: KMeansDiscretizer = None
    dataset: Any = None
    
    def __post_init__(self):
        self._num_queries = 0
        self.list_queries = {}

    @abstractmethod
    def query_oracle(self, oracle, num_queries):
        raise NotImplementedError

    def log_query(self, traj_num, idx_query):
        if traj_num not in self.list_queries:
            self.list_queries[traj_num] = set()
        self.list_queries[traj_num].add(idx_query)
        self._num_queries += 1
    
    def pop_query(self, traj_num, idx_query):
        self.list_queries[traj_num].remove(idx_query)
        self.annotated_options[traj_num][idx_query] = None
        self._num_queries -=1
    def __str__(self):
        return f'Student(num_demos={len(self.demos)},'\
                'option_dim={self.option_dim})'

    def save_queries(self, path):
        np.save(path, arr=self.list_queries)
