import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

import sys

def plot_heatmap(q_array):

    q_array_max = np.zeros((discr,discr,discr,discr))
    array= np.zeros((discr, discr, discr))
    array_2= np.zeros((discr, discr))
    for x in range(discr):
        for v in range(discr):
            for th in range(discr):
                for av in range(discr):
                    q_array_max[x,v,th,av] = max(q_array[x,v,th,av,:]) 
    for x in range(discr):
        for th in range(discr):
            for v in range(discr):
                array[x,th,v] = np.mean(q_array_max[x,v,th])
    for x in range(discr):
        for th in range(discr):
            array_2[x,th] = np.mean(array[x,th])
    plt.imshow(array_2)
    plt.colorbar()
    plt.show()

def plot_return(epl_avg):
    plt.figure()
    plt.plot(epl_avg)
    plt.title('Average Return')
    plt.xlabel('ep')
    plt.ylabel('R')
    plt.show()

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = 'TRAINING' # TRAINING, TEST
# MODE = 'TEST' 
print(MODE)

episodes = 20000
test_episodes = 200
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2
b = 2222  # TODO: choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))
# q_grid = q_grid+50

if MODE == 'TEST':
    q_grid = np.load('q_values_eps_constant.npy')
    # q_grid = np.load('q_values_eps_GLIE.npy')
    # q_grid = np.load('q_values.npy')


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av

def get_action(state, q_values, greedy=False):
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy 
        best_action_estimated = np.argmax(q_values[x,v,th,av])  # TODO: greedy w.r.t. q_grid

        return best_action_estimated

    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action
            action_chosen = np.random.choice([0,1])  # TODO: choose random action with equal probability among all actions

            return action_chosen
        else:
            # Greedy action
            best_action_estimated = np.argmax(q_values[x,v,th,av]) # TODO: greedy w.r.t. q_grid
            
            return best_action_estimated

def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma*max(q_array[new_cell_index]) # TODO

    # Update Q value
    old_q = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
    q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = old_q + alpha*(target_value - old_q)   # TODO

    return

# Training loop
ep_lengths, epl_avg = [], []

if MODE == 'TEST':
    episodes = 0

# np.save("q_values_variation_0.npy", q_grid)

for ep in range(episodes+test_episodes):
    test = ep > episodes

    if MODE == 'TEST':
        test = True

    state, done, steps = env.reset(), False, 0

    epsilon = constant_eps  # TODO: change to GLIE schedule (task 3.1) or 0 (task 3.3)
    # epsilon = 0  
    # epsilon = b/(b+ep)

    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        # else:
        #     env.render()


        
        state = new_state
        steps += 1

    # if ep == 2000:
    #     np.save("q_values_variation_middle1.npy", q_grid)
    # if ep == 5000:
    #     np.save("q_values_variation_middle2.npy", q_grid)
    # if ep == 10000:
    #     np.save("q_values_variation_middle3.npy", q_grid)

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-100):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
        print('Epsilon:', epsilon)
if MODE == 'TEST':
    print(f"TEST Average Return Value : {np.mean(ep_lengths)} average timesteps and {np.mean(ep_lengths)} {test_episodes} episodes")

# plot_heatmap(q_grid)
# plot_return(epl_avg)
# plot_return(ep_lengths)


if MODE == 'TEST':
    sys.exit()

# Save the Q-value array
np.save("q_value.npy", q_grid)
# np.save("q_values_eps_GLIE.npy", q_grid)

np.save("epl_avg", epl_avg)
# np.save("epl_avg_q_0", epl_avg)
# np.save("epl_avg_q_50", epl_avg)
# np.save("epl_avg_epsilon_GLIE.npy", epl_avg)
# np.save("epl_avg_epsilon_const.npy", epl_avg)


