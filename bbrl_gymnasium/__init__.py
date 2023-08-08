from ._version import __version__, __version_tuple__
from gymnasium.envs.registration import register

register(
    id="CartPoleContinuous-v0",
    entry_point="bbrl_gymnasium.envs:ContinuousCartPoleEnv",
    max_episode_steps=200,
)
register(
    id="CartPoleContinuous-v1",
    entry_point="bbrl_gymnasium.envs:ContinuousCartPoleEnv",
    max_episode_steps=500,
)

register(id="LineMDP-v0", entry_point="bbrl_gymnasium.envs:LineMDPEnv", max_episode_steps=100)
register(
    id="LineMDPContinuous-v0",
    entry_point="bbrl_gymnasium.envs:ContinuousLineMDPEnv",
    max_episode_steps=100,
)
register(
    id="2DMDPContinuous-v0",
    entry_point="bbrl_gymnasium.envs:Continuous2DMDPEnv",
    max_episode_steps=100,
)
register(
    id="RocketLander-v0",
    entry_point="bbrl_gymnasium.envs:RocketLanderEnv",
    max_episode_steps=1000,
    reward_threshold=0,
)
register(
    id="MazeMDP-v0",
    entry_point="bbrl_gymnasium.envs:MazeMDPEnv",
    max_episode_steps=1000
)
register(
    id="DebugV-v0",
    entry_point="bbrl_gymnasium.envs:DebugVEnv",
    max_episode_steps=10
)
register(
    id="SingleStateMDP-v0",
    entry_point="bbrl_gymnasium.envs:SingleStateMDP",
    max_episode_steps=3000
)
register(
    id="PendulumEnv-v0",
    entry_point="bbrl_gymnasium.envs:PendulumEnv",
    max_episode_steps=200
)
