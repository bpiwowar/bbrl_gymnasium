from typing import Any, Dict
import gymnasium as gym
import random
import numpy as np


class MazeMDPContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to turn the Tabular MazeMDP into a continuous state version
    """

    def __init__(self, env):
        super(MazeMDPContinuousWrapper, self).__init__(env)
        # Building a new continuous observation space from the coordinates of each state
        high = np.array(
            [
                env.coord_x.max()[0] + 1,
                env.coord_y.max()[0] + 1,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                env.coord_x.min()[0],
                env.coord_y.min()[0],
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low, high)

    def is_continuous_state():
        # By contrast with the wrapped environment where the state space is discrete
        return True

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        return self.env.reset(), {}

    def step(self, action):
        # Turn the discrete state into a pair of continuous coordinates
        # Take the coordinates of the state and add a random number to x and y to
        # sample anywhere in the [1, 1] cell...
        next_state, reward, done, info = self.env.step(action)
        x = self.env.coord_x[next_state]
        y = self.env.coord_x[next_state]
        xc = x + random.random()
        yc = y + random.random()
        next_continuous = [xc, yc]
        return next_continuous, reward, done, False, {}
