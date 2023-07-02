import logging
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
import gymnasium as gym
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)


class SingleStateMDP(gym.Env):
    def __init__(self, A0=0.3, A1=0.9, nu=5, sigma=0.25, seed=None):
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(1)

        self.state = np.zeros(1)

        self.A0 = A0
        self.A1 = A1
        self.nu = nu
        self.sigma = sigma

        self.seed()
        self.viewer = None
        self.state = None
        self.np_random = None

        self.steps_beyond_done = None

    def _mean_reward(self, action):
        A = self.A0 + (self.A1 - self.A0) * (action + 1) * 0.5
        return A * np.sin(self.nu * action)

    def step(self, action):
        reward = self._mean_reward(action) + self.rng.normal(0, self.sigma)
        reward = reward.item()
        next_state = np.zeros(1)
        return next_state, reward, False, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.state = np.zeros(1)
        return self.state], {}

    def render(self, mode="human"):
        pass
