import gymnasium as gym
import bbrl_gymnasium


def test_rocket_lander_v0():
    env = gym.make("RocketLander-v0")
    env.reset()


def test_mazemdp_v0():
    env = gym.make("MazeMDP-v0")
    env = gym.make("MazeMDP-v0", kwargs={"width": 6, "height": 5, "ratio": 0.2})
    env.reset()


def test_cartpolecontinuous_v0():
    env = gym.make("CartPoleContinuous-v0")
    env.reset()


def test_cartpolecontinuous_v1():
    env = gym.make("CartPoleContinuous-v1")
    env.reset()


def test_lineMDP_v0():
    env = gym.make("LineMDP-v0")
    env.reset()


def test_lineMDPContinuous_v0():
    env = gym.make("LineMDPContinuous-v0")
    env.reset()


def test_2DMDPContinuous_v0():
    env = gym.make("LineMDPContinuous-v0")
    env.reset()


if __name__ == "__main__":
    test_mazemdp_v0()
    test_mazemdp_v0()
