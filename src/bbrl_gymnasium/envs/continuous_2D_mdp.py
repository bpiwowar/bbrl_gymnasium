"""
Simple MDP where the state is a pair of real numbers: One in [0,1] and the second is set 0.5
The agent can perform 2 actions (left or right) which increase or decrease the state of 0.2
The agent gets a reward of 10 if it reaches 1 and a reward of 2 if it reaches 0.
"""

import logging
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)


class Continuous2DMDPEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([1, 1]))

        self.seed()
        self.viewer = None
        self.state = None
        self.np_random = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: ActType) -> ObsType:
        done = False
        reward = 0.0
        if action == 0:
            self.state[0] += 0.2
            if self.state[0] >= 1:
                done = True
                reward = 10.0
        else:
            self.state[0] -= 0.2
            if self.state[0] < 0:
                done = True
                reward = 2.0

        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior."
                )
                self.steps_beyond_done += 1
        next_state = np.array(self.state)
        return next_state, reward, done, False, {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self.state = [0.4, 0.5]
        self.steps_beyond_done = None
        return self.state, {}

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
        print("Nothing to show")
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
