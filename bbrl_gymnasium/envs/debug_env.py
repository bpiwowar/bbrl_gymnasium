"""
Simple MDP where the state is a real number in [0,1].
Any action increases the state of 0.1
The agent starts at 0.01 and gets a reward of 1 for any movement, until it moves out
"""

import logging
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)


class DebugVEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            np.array([0]), np.array([1]), dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.np_random = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reward = 1.0
        self.state += 0.1
        if self.state >= 1:
            done = True
        next_state = np.array([self.state])
        return next_state, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):

        self.state = 0.01
        return np.array([self.state]), {}

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
