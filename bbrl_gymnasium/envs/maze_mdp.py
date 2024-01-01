"""
Simple Maze MDP
"""

from functools import partial
import logging
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from tabularmazemdp import create_random_maze
from tabularmazemdp.maze import build_maze

logger = logging.getLogger(__name__)


class MazeMDPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "video.frames_per_second": 5}

    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode

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
        self.state = self.mdp.current_state # should not be used for learning (reset-anywhere property)

        self.seed()
        self.np_random = None
        self.title = f"Simple maze {width}x{height}"

        self.set_render_func(partial(self.init_draw, self.title))

    def set_title(self, title):
        self.title = title

    def set_render_func(self, render_func: Callable, *args, **kwargs):
        """Sets the render mode"""
        kwargs = {**kwargs, "mode": self.render_mode}
        self.render_func = partial(render_func, *args, **kwargs)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.mdp.step(action)
        return next_state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        r = self.mdp.reset(**kwargs)
        if isinstance(r, list):
            return r
        return self.mdp.reset(**kwargs), {}

    def sample_transition(self, **kwargs):
        state, action, next_state = self.mdp.sample_transition(**kwargs)
        return state, action, next_state

    def _draw(self, recorder, callable, *args, **kwargs):
        """Draw and record 

        :param recorder: A video recorder (or None)
        """
        if recorder is not None:
            if recorder.enabled:
                self.set_render_func(callable, *args, **kwargs)
                recorder.capture_frame()
        else:
            return callable(*args, **kwargs)

    # Drawing functions
    def draw_v_pi_a(self, v, policy, agent_pos, title="MDP studies", mode="legacy", recorder=None):
        return self._draw(recorder, self.render, v, policy, agent_pos, title=title, mode=mode)

    def draw_v_pi(self, v, policy, title="MDP studies", mode="legacy", recorder=None):
        return self._draw(recorder, self.mdp.render, v, policy, None, title, mode=mode)

    def draw_v(self, v, mode="legacy", title="MDP studies", recorder=None):
        return self._draw(recorder, self.mdp.render, v, None, None, title, mode=mode)

    def draw_pi(self, policy, title="MDP studies", mode="legacy", recorder=None):
        return self._draw(recorder, self.mdp.render, None, policy, None, title, mode=mode)

    def init_draw(self, title, mode="legacy", recorder=None):
        return self._draw(recorder, self.mdp.new_render, title, mode=mode)

    def render(self):
        return self.render_func(mode=self.render_mode)

    def render_mdp(self, func, *args, **kwargs):
        self.set_render_func = partial(func, *args, **kwargs)
        self.render()

    def set_no_agent(self):
        self.mdp.has_state = False

    def set_timeout(self, timeout):
        self.mdp.timeout = timeout
