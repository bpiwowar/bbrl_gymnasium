import gymnasium as gym
import bbrl_gymnasium
import numpy as np
from typing import Tuple, List


from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv


def get_policy_from_v(mdp: MazeMDPEnv, v: np.ndarray) -> np.ndarray:
    # Outputs a policy given the state values
    policy = np.zeros(mdp.nb_states)  # initial state values are set to 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        if x not in mdp.terminal_states:
            for u in range(mdp.action_space.start, mdp.action_space.start + mdp.action_space.n):
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
            policy[x] = np.argmax(v_temp)
    return policy


# ------------------------- Value Iteration with the V function ----------------------------#
def value_iteration_v(
    mdp: MazeMDPEnv, render: bool
) -> Tuple[np.ndarray, List[float]]:
    # Value Iteration using the state value v
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    v_list = []
    stop = False

    while not stop:
        v_old = v.copy()
        if render:
            mdp.draw_v(v, title="Value iteration V", mode="rgb_array")

        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of the state x for each action u of the MDP action space
            if x not in mdp.terminal_states:
                v_temp = []
                for u in range(mdp.action_space.start, mdp.action_space.start + mdp.action_space.n):
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                    v_temp.append(mdp.r[x, u] + mdp.gamma * summ)

                # Select the highest state value among those computed
                v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True
        v_list.append(np.linalg.norm(v))

    if render:
        policy = get_policy_from_v(mdp, v)
        mdp.draw_v_pi(v, policy, title="Value iteration V", mode="rgb_array")

    return v, v_list


def test_mazemdp_v0():
    env = gym.make("MazeMDP-v0", kwargs={"width": 6, "height": 5, "ratio": 0.2}, render="rgba")
    env.reset()
    # env.set_no_agent()
    v, v_list = value_iteration_v(env, render=True)


if __name__ == "__main__":
    test_mazemdp_v0()
