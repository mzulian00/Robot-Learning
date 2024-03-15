"""
    Robot Learning
    Exercise 2

    Linear Quadratic Regulator

    Polito A-Y 2023-2024
"""
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    a = g/(lp*(4.0/3 - mp/(mp+mk)))
    # state matrix
    A = np.array([[0, 1, 0, 0],
                [0, 0, a, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = -1/(lp*(4.0/3 - mp/(mp+mk)))
    B = np.array([[0], [1/mt], [0], [b]])
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """
    #TODO: 
    # ...
    R_values = [0.01, 0.1, 10, 100]
    x_R = []
    for R in R_values:
        x_R.append( control(env, mp, mk, l, g, time_sleep, terminate,R) )

    plot_states_R(x_R,R_values)
    
    return


def control(env, mp, mk, l, g, time_sleep=False, terminate=True, R=1):
    """
    Control using LQR
    """
    #TODO: plot the states of the system ...

    obs = env.reset()    # Reset the environment for a new episode
    x_t = [obs]
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B, R)    # Re-compute the optimal controller for the current R value
    negative_force_flag = False
    for i in range(1000):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)

        if i==0 and force < 0:
            negative_force_flag = True
        if negative_force_flag:
            x_t.append(-obs)
        else:
            x_t.append(obs)
        if terminate and done and i==399:
            print(f'Terminated after {i+1} iterations.')
            break
    if R == 1:
        plot_states(x_t)

    return x_t
    


# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        x_t = control(env, mp, mk, l, g, args.time_sleep, terminate=True)

        times = [-1, -1, -1, -1]
        for i in range(x_t.__len__()):
            for t in range(4):
                if abs(x_t[i][t]) < 0.05 and times[t] < 0:
                    times[t] = i
                if abs(x_t[i][t]) >= 0.05 and times[t] >= 0:
                    times[t] = -1
        print(f'Convergence time x1 = {times[0]}')
        print(f'Convergence time x2 = {times[1]}')
        print(f'Convergence time x3 = {times[2]}')
        print(f'Convergence time x4 = {times[3]}')

    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)
   
    

    env.close()


def plot_states(x_t):
    plt.figure(figsize=(10, 6))
    red_line = [0.05 for x in x_t]
    red_line_n = [-0.05 for x in x_t]
    plt.subplot(2, 2, 1) 
    plt.plot([x[0] for x in x_t])
    plt.plot(red_line, 'r')
    plt.plot(red_line_n, 'r')
    plt.title(f'x')

    plt.subplot(2, 2, 2) 
    plt.plot([x[1] for x in x_t])
    plt.plot(red_line, 'r')
    plt.plot(red_line_n, 'r')
    plt.title(f'x_dot')

    plt.subplot(2, 2, 3) 
    plt.plot([x[2] for x in x_t])
    plt.plot(red_line, 'r')
    plt.plot(red_line_n, 'r')
    plt.title(f'theta')

    plt.subplot(2, 2, 4)  
    plt.plot([x[3] for x in x_t])
    plt.plot(red_line, 'r')
    plt.plot(red_line_n, 'r')
    plt.title(f'theta_dot')
  
    plt.tight_layout()
    plt.show()
    return

def plot_states_R(x_R,R_values):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 2, 1)     
    for i in range(R_values.__len__()):
        plt.plot([x[0] for x in x_R[i]],label=f'R={R_values[i]}')
    plt.title(f'x')
    plt.legend()

    plt.subplot(2, 2, 2)     
    for i in range(R_values.__len__()):
        plt.plot([x[1] for x in x_R[i]],label=f'R={R_values[i]}')
    plt.title(f'x_dot')
    plt.legend()

    plt.subplot(2, 2, 3)     
    for i in range(R_values.__len__()):
        plt.plot([x[2] for x in x_R[i]],label=f'R={R_values[i]}')
    plt.title(f'theta')
    plt.legend()

    plt.subplot(2, 2, 4)     
    for i in range(R_values.__len__()):
        plt.plot([x[3] for x in x_R[i]],label=f'R={R_values[i]}')
    plt.title(f'theta_dot')
    plt.legend()

    plt.show()
    return


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)


#times
# 0+-0.05 from x0=0 
# t1 = 0
# t2 = 28
# t3 = 0
# t4 = 10


