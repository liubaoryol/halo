import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import List


class RescueWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, n_targets=1,
                 latent_distribution=lambda x: 0,
                 allow_variable_horizon=True,
                 fixed_targets=None,
                 obstacles=None,
                 danger=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._max_episode_steps = n_targets**2 * size**3

        self.n_targets = n_targets
        self.obstacles = obstacles
        self.danger = danger
        self.fixed_targets = fixed_targets
        self.occupied_grids = np.empty((n_targets + 1, 2), dtype=np.int64)
        self.latent_distribution = latent_distribution

        self.allow_variable_horizon=allow_variable_horizon
        self.at_absorb_state = False

        obs_shape = 2*(n_targets + 1)
        
        highs = [ size-1 ] * obs_shape
        highs += [ 1 ] * n_targets
        highs.append(n_targets - 1)

        self.observation_space = spaces.Box(
            0, np.array(highs), shape=(obs_shape + n_targets + 1,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps actions from `self.action_space` to 
        the direction.
        E.g. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.danger_reward = -100
        self.obstacle_reward = -1
        self.target_reward = 1000

    @property
    def danger(self):
        return self._danger
    
    @property
    def obstacles(self):
        return self._obstacles

    @property
    def fixed_targets(self):
        return self._fixed_targets

    @danger.setter
    def danger(self, value):
        self._danger = -np.ones((1,2)) if value is None else np.array(value)
        assert self._danger.shape[1]==2, f'Danger grids must have shape \
                                          (N, 2); instead they are shape \
                                          {self._danger.shape}'

    @obstacles.setter
    def obstacles(self, value):
        if value is None:
            self._obstacles = -np.ones((1,2))
        else:
            self._obstacles = np.array(value)

        assert self._obstacles.shape[1]==2, f'Obstacles must have shape \
                                            (N, 2); instead they are shape \
                                            {self._obstacles.shape}'

    @fixed_targets.setter
    def fixed_targets(self, targets):
        self._fixed_targets = targets

        if targets is not None:
            assert len(targets)==self.n_targets, \
            'Number of targets should be {}, but {} were given'.format(
                self.n_targets, len(targets))
        
            self._fixed_targets = np.array(targets)

            
            assert self._fixed_targets.shape[1]==2, f'Obstacles must have \
                                                    shape (N, 2); instead \
                                                    they are shape \
                                                    {self._fixed_targets.shape}'
            for target in targets:
                assert not self._in_obstacle(target), \
                    'Targets must be outside obstacles'
                assert (np.clip(target, 0, self.size - 1) == target).all(), \
                    'Targets must be inside grid'

    def get_obs(self):
        visited = [0 if t in self._visited_goals else 1 for t in range(self.n_targets)]

        obs = np.concatenate(self.occupied_grids)
        obs = np.concatenate([obs, visited, [self._curr_goal]])

        return obs

    def _is_occupied(self, location):
        A = self._in_obstacle(location)
        B = (self.occupied_grids==location).all(axis=1).any()
        return (A or B)

    def _in_obstacle(self, location):
        return (self.obstacles==location).all(axis=1).any()

    def _in_danger(self, location):
        return (self.danger==location).all(axis=1).any()

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, targets=self.fixed_targets)

    def _reset(self, seed=None, targets=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.at_absorb_state = False

        self.occupied_grids = np.empty((self.n_targets + 1, 2), dtype=np.int64)
        self._elapsed_steps = 0
        self._visited_goals = []
        self._curr_goal = self.sample_next_goal()

        if targets is not None:
            assert len(targets)==self.n_targets, \
            'Number of targets should be {}, but {} were given'.format(
                self.n_targets, len(targets))
            self.occupied_grids[1:] = targets
            
        else:
            # We will sample the target's location randomly until it does not coincide with the agent's location
            for element in range(1, self.n_targets+1):
                target_location = self.occupied_grids[0]
                while self._is_occupied(target_location):
                    target_location = self.np_random.integers(
                        0, self.size, size=2, dtype=int
                    )
                self.occupied_grids[element] = target_location

        # Choose the agent's location uniformly at random
        agent_location = self.np_random.integers(0, self.size, size=2)
        while self._is_occupied(agent_location):
            agent_location = self.np_random.integers(0, self.size, size=2)

        self.occupied_grids[0] = agent_location


        observation = self.get_obs()
        info = {}

        if self.render_mode=="human":
            self._render_frame()
        elif self.render_mode=="rgb_array":
            observation = self._render_frame()

        return observation, info

    def step(self, action):
        truncated = False
        terminated = False

        self._elapsed_steps += 1

        if self.at_absorb_state:
            reward = 0

        else:
            if len(self._visited_goals) == self.n_targets:
                raise ValueError("All targets have been visited. Call `reset()` function")

            reward = -1
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            agent_location = self.occupied_grids[0]
            agent_location, reward = self.move_agent(agent_location, direction)
            self.occupied_grids[0] = agent_location

            curr_target = self.occupied_grids[self._curr_goal + 1]
            # An episode is done iff the agent has reached the target
            if self.target_achieved(agent_location, curr_target):
                self._visited_goals.append(self._curr_goal)
                reward = self.target_reward
                if len(self._visited_goals) == self.n_targets:
                    if self.allow_variable_horizon:
                        terminated = True
                    else:
                        self.at_absorb_state = True
                else:
                    self._curr_goal = self.sample_next_goal()
            elif self._in_danger(agent_location):
                reward = self.danger_reward
        
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            terminated = True

        observation = self.get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode=="rgb_array":
            observation = self._render_frame()

        return observation, reward, terminated, truncated, info

    def move_agent(self, agent_location, direction):
        new_location = np.clip(agent_location + direction, 0, self.size - 1)
        if (new_location==self.obstacles).all(axis=1).any():
            return agent_location, self.obstacle_reward
        else:
            return new_location, -1
            
    def target_achieved(self, agent_location, target):
        return np.array_equal(agent_location, target)

    def step_from_obs(self, obs, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        agent_location = obs[:2]
        agent_location = np.clip(
            agent_location + direction, 0, self.size - 1
        )
        return np.concatenate([agent_location, obs[2:]])

    def sample_next_goal(self):
        targets = list(range(self.n_targets))
        for visited in self._visited_goals:
            targets.remove(visited)
        m = len(targets)
        # if m > 0
        return targets[self.latent_distribution(m)]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        agent_location = self.occupied_grids[0]
        targets = self.occupied_grids[1:]
        for idx, target_location in enumerate(targets):
            if idx not in self._visited_goals:
                color = (155, 0, 0) if idx==self._curr_goal else (255, 0, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        pix_square_size * target_location,
                        (pix_square_size, pix_square_size),
                    ),
                )
        
        # Draw obstacles
        for obstacle in self.obstacles:
            color = (0, 0, 0)
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Draw danger
        for danger in self.danger:
            color = (255,211,67)
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    pix_square_size * danger,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(2, 1, 0)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def visual_interactive(self):
        assert self.render_mode == 'human', "`render_mode` should be 'human'"
        
        observation, info = self.reset()

        while True:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 0
                    elif event.key == pygame.K_UP:
                        action = 3
                    elif event.key == pygame.K_DOWN:
                        action = 1
                        
                    observation, reward, terminated, truncated, info = self.step(action)
                    print("Reward: ", reward)
                    if terminated or truncated:
                        observation, info = self.reset()


class FilterLatent(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, unobservable_states: List):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.utils.RecordConstructorArgs.__init__(self, unobservable_states=unobservable_states)
        gym.ObservationWrapper.__init__(self, env)


        wrapped_observation_space = env.observation_space
        if not isinstance(wrapped_observation_space, spaces.Box):
            raise ValueError(
                f"FilterLatent is only usable with Box observations, "
                f"environment observation space is {type(wrapped_observation_space)}"
            )

        self._unobservable_states = unobservable_states
        highs = wrapped_observation_space.high
        lows = wrapped_observation_space.low
        

        self.mask = np.full(len(highs), True, dtype=bool)
        self.mask[self._unobservable_states] = False


        self.observation_space = type(wrapped_observation_space)(
            lows[self.mask], highs[self.mask]
        )

        self._env = env


    def observation(self, observation):
        return observation[self.mask]

