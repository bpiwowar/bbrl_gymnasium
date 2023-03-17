"""
Simple Maze MDP
"""

import logging
from typing import Callable

try:
    # Necessary to handle action space from gym...
    import gym as old_gym
except:
    pass

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from mazemdp import create_random_maze
from mazemdp.maze import build_maze

logger = logging.getLogger(__name__)


class MazeMDPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "video.frames_per_second": 5}

    def __init__(self, **kwargs):
        if kwargs == {}:
            width = 10
            height = 10
            self.mdp, nb_states, coord_x, coord_y = create_random_maze(10, 10, 0.2)
        else:
            kwargs = kwargs["kwargs"]
            width = kwargs["width"]
            height = kwargs["height"]
            if "hit" not in kwargs.keys():
                hit = False
            else:
                hit = kwargs["hit"]
            if "walls" not in kwargs.keys():
                ratio = kwargs["ratio"]
                self.mdp, nb_states, coord_x, coord_y = create_random_maze(
                    width, height, ratio, hit
                )
            else:
                self.mdp, nb_states, coord_x, coord_y = build_maze(
                    width, height, kwargs["walls"], hit
                )

        self.nb_states = nb_states
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.observation_space = spaces.Discrete(nb_states)
        self.action_space = spaces.Discrete(4)
        self.terminal_states = [nb_states - 1]
        self.P = self.mdp.P
        self.gamma = self.mdp.gamma
        self.r = self.mdp.r

        self.seed()
        self.np_random = None
        self.title = f"Simple maze {width}x{height}"

        self.set_render_func(self.init_draw, lambda draw: draw(self.title))

    def set_title(self, title):
        self.title = title

    def set_render_func(self, render_func: Callable, callable: Callable):
        """Sets the render mode"""

        def call(mode: str):
            def draw(*args, **kwargs):
                return render_func(*args, **kwargs, mode=mode)

            return callable(draw)

        self.render_func = call

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.mdp.step(action)

    def reset(self, **kwargs):
        r = self.mdp.reset(**kwargs)
        if isinstance(r, list):
            return r
        return self.mdp.reset(**kwargs), {}

    # Drawing functions
    def draw_v_pi_a(self, v, policy, agent_pos, title="MDP studies", mode="legacy"):
        return self.mdp.render(v, policy, agent_pos, title, mode=mode)

    def draw_v_pi(self, v, policy, title="MDP studies", mode="legacy"):
        agent_pos = None
        return self.mdp.render(v, policy, agent_pos, title, mode=mode)

    def draw_v(self, v, mode="legacy", title="MDP studies"):
        policy = None
        agent_pos = None
        return self.mdp.render(v, policy, agent_pos, title, mode=mode)

    def draw_pi(self, policy, title="MDP studies", mode="legacy"):
        v = None
        agent_pos = None
        return self.mdp.render(v, policy, agent_pos, title, mode=mode)

    def init_draw(self, title, mode="legacy"):
        return self.mdp.new_render(title, mode=mode)

    def render(self, mode="human"):
        r = self.render_func(mode)
        return r

    def set_no_agent(self):
        self.mdp.has_state = False

    def set_timeout(self, timeout):
        self.mdp.timeout = timeout
