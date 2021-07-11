import random
from copy import deepcopy

import numpy as np
import os
from finite_env import FiniteEnv

# modified from https://github.com/yfletberliac/rlss-2019/blob/master/utils/frozen_lake.py

class MDP(FiniteEnv):
    """
    Environment with 3 states and 2 actions per state that gives a reward of 1 when going to the
    last state and 0 otherwise.

    Args:
        seed    (int): Random number generator seed

    """

    def __init__(self, P, bad_states=[], seed=42):
        # Set seed
        self.RS = np.random.RandomState(seed)

        # Transition probabilities
        # shape (Ns, Na, Ns)
        # P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a)

        P = P
        Ns, Na, _ = P.shape
        
        self.Ns = Ns
        self.Na = Na
        self.bad_states = set(bad_states)
        self.action_space = list(range(Na))

        # Initialize base class
        states = np.arange(Ns).tolist()
        action_sets = [np.arange(Na).tolist()]*Ns
        super().__init__(states, action_sets, P)

    def reward_func(self, state, action, next_state):
        if next_state not in self.bad_states:
            return 1.0 * (next_state == self.Ns - 1)
        elif next_state in self.bad_states:
            return (-10.0)

    def reset(self, s=3):
        self.state = s
        return self.state

    def step(self, action):

        next_state = self.sample_transition(self.state, action)
        reward = self.reward_func(self.state, action, next_state)
        
        if self.state in self.bad_states or self.state == self.Ns-1:
            done = True
        else:
            done = False
            
        info = {}
        self.state = next_state

        observation = next_state
        return observation, reward, done, info

    def sample_transition(self, s, a):
        prob = self.P[s, a, :]
        s_ = self.RS.choice(self.states, p = prob)
        return s_
    
    def render(self):        
        
        env_to_print = ""

        
        for state in range(0,self.Ns-1) :
            if state % 4 == 0:
                env_to_print += "\n"
            
            if state in self.bad_states:
                env_to_print += "H"
            elif state not in self.bad_states:
                if state == 3:
                    env_to_print += "S"
                else:
                    env_to_print += "F"
                    
        env_to_print += "G"
        
        print("(S: starting point, safe) (F: frozen surface, safe) (H: hole, fall to your doom) (G: goal)")
        print("(H: -10 reward) (G: +1 reward)")
        print("=================")
        print(env_to_print)
        print("=================")
        print("Current state", self.state)
        
        
class FrozenLake(MDP):
    def __init__(self, data_path="./data"):
        self.data_path = data_path
        P = np.load(os.path.join(data_path, "det_trans_matrix.npy"))
        bad_states = [7, 11]
        super().__init__(P=P, bad_states=bad_states)

    def change_trans_prob(self, t_a=None):
        det_P = np.load(os.path.join(self.data_path, "det_trans_matrix.npy"))
        if t_a is None:
            t_a = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        tmp = np.zeros(det_P.shape[2])
        new_P = np.zeros_like(det_P)
        for i in range(det_P.shape[0]):
            for j in range(det_P.shape[1]):
                for j2 in range(det_P.shape[1]):
                    for j3 in range(det_P.shape[2]):
                        if det_P[i, j2, j3] == 1:
                            tmp[j3] += det_P[i, j2, j3] * t_a[j][j2]

                new_P[i, j] = deepcopy(tmp)
                tmp = np.zeros(det_P.shape[2])

        self.P = new_P
