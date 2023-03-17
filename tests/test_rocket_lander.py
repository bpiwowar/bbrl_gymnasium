import gymnasium as gym
import bbrl_gymnasium

env_id = "RocketLander-v0"
rl_env = gym.make(env_id)
obs, info = rl_env.reset()
print(obs.shape)
