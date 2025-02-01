__credits__ = ["Rushiv Arora"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {}


def nb_bodies_from_xml_file_name(xml_file):
    fields = xml_file.split(".")
    tmp = fields[-2]
    swimm_name = tmp.split("/")
    s_name = swimm_name[-1]
    num = s_name[7:]
    if num == "":
        return 3
    retour = int(num)
    return retour


class SwimmerBBRLEnv(SwimmerEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        xml_file="swimmer.xml",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        nb_bodies = nb_bodies_from_xml_file_name(xml_file)
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(nb_bodies * 2 + 2,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(nb_bodies * 2 + 4,), dtype=np.float64
            )
        MujocoEnv.__init__(
            self, xml_file, 4, observation_space=observation_space, **kwargs
        )
