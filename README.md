The bbrl_gymnasium library is the place where I put additional gym-like environments.

So far, it contains the following environments:
- CartPoleContinuous-v0 (with timit limit = 200 steps)
- CartPoleContinuous-v1 (with timit limit = 500 steps)
- MazeMDP, a Maze environment for Tabular RL
- RocketLander-v0, a rocket landing simulation adapted from [this repository](https://github.com/sdsubhajitdas/Rocket_Lander_Gym)
- LineMDP-v0, a simple discrete state and action MDP
- LineMDPContinuous-v0, a simple discrete action MDP
- 2DMDPContinuous-v0, a discrete action MDP with 2D state


## Installation

```
pip install bbrl_gymnasium
```

## Use it

```
import gymnasium as gym
import bbrl_gymnasium

env = gym.make("CartPoleContinuous-v0")  # or -v1 or any other and then use your environment as usual
```
