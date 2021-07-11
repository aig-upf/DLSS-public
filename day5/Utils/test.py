from copy import deepcopy

from matplotlib import pyplot as plt

from day5.Utils.policy_utils import state_visited_policy
from frozen_lake import FrozenLake
import numpy as np
env = FrozenLake(data_path="/home/lorenzo/Documenti/UPF/summerschool/DLSS/day5/Utils")
probability_selected_a = 0.9 #@param {type: "slider", "min":0.3, "max": 1, "step": 0.1}
probability_selected_a = float(probability_selected_a)
probability_rem_a = float((1 - probability_selected_a)/3)
prob_a = [[probability_selected_a, probability_rem_a, probability_rem_a, probability_rem_a],
          [probability_rem_a, probability_selected_a, probability_rem_a, probability_rem_a],
          [probability_rem_a, probability_rem_a, probability_selected_a, probability_rem_a],
          [probability_rem_a, probability_rem_a, probability_rem_a, probability_selected_a]]
env.change_trans_prob(prob_a)

env.render()
np.random.seed(42)

# 0 = sinistra
# 1 = giu
# 2 = destra
# 3 = su

# for i in range(env.P.shape[0]):
#     print(env.P[i])

# print(env.P[5])
#
# env.P[5] = np.zeros(shape=(env.Na, env.Ns))
#
# print(env.P[5])
# print(env.P[6])
#
# env.P[5, 0, 4] = 1
# env.P[5, 1, 9] = 1
# env.P[5, 2, 6] = 1
# env.P[5, 3, 1] = 1
#
# print("")
#
# print(env.P[12])
#
# env.P[12] = np.zeros(shape=(env.Na, env.Ns))
#
# print(env.P[12])
# print(env.P[13])
#
# env.P[12, 0, 12] = 1
# env.P[12, 1, 12] = 1
# env.P[12, 2, 13] = 1
# env.P[12, 3, 8] = 1







def bellman_operator(Q, env, gamma=0.95):
    TQ = np.zeros((env.Ns, env.Na))
    greedy_policy = np.zeros(env.Ns)
    for s in env.states:
        for a in env.actions:
            prob = env.P[s, a, :]
            rewards = np.array([float(env.reward_func(s, a, s_)) for s_ in env.states])
            TQ[s, a] = np.sum(prob * (rewards + gamma * Q.max(axis=1)))

    greedy_policy = np.argmax(TQ, axis=1)

    return TQ, greedy_policy

def value_iteration(Q0, env, epsilon=1e-5):
    Q = Q0
    while True:
        TQ, greedy_policy = bellman_operator(Q, env, gamma = 0.95)

        err = np.abs(TQ-Q).max()
        if err < epsilon:
            return TQ, greedy_policy

        Q = TQ


# -------------------------------
# Q-Learning implementation
# ------------------------------

class QLearning:
    """
    Implements Q-learning algorithm with epsilon-greedy exploration

    If learning_rate is None; alpha(x,a) = 1/max(1, N(s,a))**alpha
    """

    def __init__(self, env, gamma=0.95, alpha=0.6, learning_rate=None, min_learning_rate=0.01, epsilon=1.0,
                 epsilon_decay=0.9995,
                 epsilon_min=0.25, seed=42):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((env.Ns, env.Na))
        self.Nsa = np.zeros((env.Ns, env.Na))
        self.state = env.reset()
        self.RS = np.random.RandomState(seed)

    def get_delta(self, r, x, a, y, done):
        """
        :param r: reward
        :param x: current state
        :param a: current action
        :param y: next state
        :param done:
        :return:
        """
        max_q_y_a = self.Q[y, :].max()
        q_x_a = self.Q[x, a]

        return r + self.gamma * max_q_y_a - q_x_a

    def get_learning_rate(self, s, a):
        if self.learning_rate is None:
            return max(1.0 / max(1.0, self.Nsa[s, a]) ** self.alpha, self.min_learning_rate)
        else:
            return max(self.learning_rate, self.min_learning_rate)

    def get_action(self, state):
        if self.RS.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(self.env.actions)
        else:
            # exploit
            a = self.Q[state, :].argmax()
            return a

    def step(self):
        # Current state
        x = self.env.state

        # Choose action
        a = self.get_action(x)

        # Learning rate
        alpha = self.get_learning_rate(x, a)

        # Take step
        observation, reward, done, info = self.env.step(a)
        y = observation
        r = reward
        delta = self.get_delta(r, x, a, y, done)

        # Update
        self.Q[x, a] = self.Q[x, a] + alpha * delta  ### WRITE YOUR CODE HERE

        self.Nsa[x, a] += 1

        if done:
            # print(x, observation, reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.env.reset()

# ---------------------------
# Convergence of Q-Learning
# ---------------------------

# Number of Q learning iterations
n_steps = int(3e5)

# Get optimal value function and its greedy policy
Q0 = np.zeros((env.Ns, env.Na))
Q_opt, pi_opt = value_iteration(Q0, env, epsilon=1e-6)

# Create qlearning object
qlearning = QLearning(env, gamma=0.95, learning_rate=0.3)

# Iterate
tt = 0
Q_est = np.zeros((n_steps, env.Ns, env.Na))
while tt < n_steps:
    qlearning.step()
    # Store estimate of Q*
    Q_est[tt, :, :] = qlearning.Q
    tt +=1


# Compute greedy policy (with estimated Q)
greedy_policy = np.argmax(qlearning.Q, axis=1)

print(env.render())

for state in env.states:
    print("state:", state)
    print("true: ", Q_opt[state, :])
    print("est: ", Q_est[-1, state, :])
    print("----------------------------")

print("optimal policy: ", pi_opt)
state_visited_opt = state_visited_policy(pi_opt)
plt.imshow(state_visited_opt.reshape((4,4)), cmap="Greys")
plt.title("Optimal policy \n ( The states in black represent the states selected by the greedy optimal policy")
plt.show()
print("est policy:", greedy_policy)
state_visited_est = state_visited_policy(greedy_policy)
plt.imshow(state_visited_est.reshape((4,4)), cmap="Greys")
plt.title("Estimated policy \n ( The states in black represent the states selected by the greedy estimated policy")
plt.show()

# Plot
diff = np.abs(Q_est - Q_opt).mean(axis=(1,2))
plt.plot(diff)
plt.xlabel('iteration')
plt.ylabel('Error')
plt.title("Q-learning convergence")
plt.show()


