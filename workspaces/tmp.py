import gym
import envs
from pathlib import Path
from gym.wrappers import RecordVideo


env = gym.make('kitchen-all-v0')
env = RecordVideo(
        env,
        video_folder=Path.cwd(),
        episode_trigger=lambda x: x % 1 == 0,
    )
env = gym.wrappers.FlattenObservation(self.env)
env.reset()

for i in range(100):
    env.step(env.action_space.sample())

env.close()