from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.benchmark import BENCHMARK_MAPPING
from libero.libero import get_default_path_dict
import os
import numpy as np
import gym
# os.environ["MUJOCO_GL"] = "disable"
# os.environ["DISPLAY"] = ""

task_order = 0 # can be from {0 .. 21}, default to 0, which is [task 0, 1, 2 ...]
benchmark_instance = BENCHMARK_MAPPING['libero_goal'](task_order)
env_args = {'camera_heights': 128,
 'camera_widths': 128,
 'has_renderer': False,
 'has_offscreen_renderer': False, 
 'use_camera_obs': False }

# dataset_path = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i))
# task_i_dataset, shape_meta = get_dataset(
#         dataset_path=dataset_path,
#         obs_modality=modality,
#         initialize_obs_utils=(i==0),
#         seq_len=1,
# )

class ConcatControlEnv(gym.Env):
    """Environment containing libero-goal tasks (10)
    
    """
    def __init__(self, env_args=env_args):

        super().__init__()
        dataset_path = os.path.join(get_default_path_dict()['bddl_files'], 'libero_goal')
        bddl_files = benchmark_instance.get_task_bddl_files()
        self.curr_subtask = 0
        bddl = bddl_files[0]
        env_args['bddl_file_name'] = os.path.join(dataset_path, bddl)
        self.render_env = ControlEnv(**env_args)
        self.envs = []
        self.ALL_TASKS = [bddl[:-5]]

        env_args['has_renderer'] = False
        for bddl in bddl_files:
            env_args['bddl_file_name'] = os.path.join(dataset_path, bddl)
            env = ControlEnv(**env_args)
            self.envs.append(env)
            self.ALL_TASKS.append(bddl[:-5])
        self.tasks_to_complete = self.ALL_TASKS
        self.spec = gym.envs.registration.EnvSpec(
            id="libero-goal-v0",
            entry_point="envs.libero.libero_goal:ConcatControlEnv",
            max_episode_steps=500,
            reward_threshold=1.0,
        )
        self.action_space = gym.spaces.Box(
           low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
           low=-np.inf, high=np.inf, shape=(79,), dtype=np.float32)
        # env_args['has_renderer'] = True

        # Initialize environments
        self.reset()
    @property
    def env(self):
        return self.curr_env
    
    @property
    def unwrapped(self):
        return self.curr_env
    
    def reset(self):
        for env in self.envs:
            env.reset()
        self.render_env.reset()
        return self.curr_env.get_sim_state()
    
    def step(self, action):
        reward = 0
        done = False
        for env in self.envs:
            out = env.step(action)
            reward += out[1]
            done = done or out[2]
        out = self.curr_env.get_sim_state(), reward, done, {}
        return out
    
    def render(self, mode="human"):
        if mode=="human":
            st = self.curr_env.get_sim_state()
            self.render_env.set_init_state(st)
            self.render_env.env.render()

    def close_renderer(self):
        self.render_env.env.close_renderer()
        
    def change_subtask(self, new_subtask):
        curr_obs = self.curr_env.get_sim_state()

        self.curr_subtask = new_subtask
        self.curr_env.reset()
        self.curr_env.set_init_state(curr_obs)

    def reset_from_known(self):
        init_states = benchmark_instance.get_task_init_states(self.curr_subtask)
        init_state_id = np.random.choice(range(50))
        for env in self.envs:
            env.set_init_state(init_states[init_state_id])
        # self.curr_env.set_init_state(init_states[init_state_id])
        return init_states[init_state_id]
        
    @property
    def curr_env(self):
        return self.envs[self.curr_subtask]