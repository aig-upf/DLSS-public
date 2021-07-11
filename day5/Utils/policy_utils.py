from frozen_lake import FrozenLake
import numpy as np

def state_visited_policy(policy):
    det_tmp_env = FrozenLake(data_path="./DLSS/day5/Utils")

    state = det_tmp_env.reset()
    states_visited = np.zeros((16))
    states_visited[state] = 1
    n_steps = 0
    while True:
        n_steps += 1
        action = policy[state]
        next_state, reward, done, info = det_tmp_env.step(action)
        states_visited[next_state] = 1
        if done:
            break
        else:
            state = next_state

        if n_steps > 100:
            break

    return states_visited