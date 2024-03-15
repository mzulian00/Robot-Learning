import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from agent import Agent, Policy
from cp_cont import CartPoleEnv
import pandas as pd

import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)

def create_model(args, env, log_dir="./tmp/gym/"):
    # T4 TODO

    if args.algo == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=args.lr, tensorboard_log=log_dir)
    elif args.algo == 'sac':
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=args.lr, tensorboard_log=log_dir)
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model
        
def load_model(args, env):
    # T4 TODO
    if args.algo == 'ppo':
        model = PPO.load("ppo_cartpole")
    elif args.algo == 'sac':
        model = SAC.load("sac_cartpole")
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return model

def save_model(args, model):
    # T4 TODO
    if args.algo == 'ppo':
        model.save("ppo_cartpole")
    elif args.algo == 'sac':
        model.save("sac_cartpole")
    else:
        raise ValueError(f"RL Algo not supported: {args.algo}")
    return

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="ContinuousCartPole-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="The total number of samples to train on")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make(args.env)

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the model from the file and go directly to testing.
    if args.test is None:
        try:
            model = create_model(args, env, log_dir)
            # Policy training (T4) TODO
            model.learn(total_timesteps=args.total_timesteps)
            # Saving model (T4) TODO
            save_model(args, model)
            plot_results(log_dir)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        print("Testing...")
        model = load_model(args, env)

        tot_rewards = np.zeros((args.test_episodes,))
        for ep in range(args.test_episodes):
            obs = env.reset()
            done = False

            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                tot_rewards[ep] += rewards
                if args.render_test == True:
                    env.render()
           
        # Policy evaluation (T4) TODO
        mean_reward, std_reward = np.mean(tot_rewards), np.std(tot_rewards)
        

        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

    env.close()    
# python -m tensorboard.main --logdir=./tmp/gym/
# PPO 100k Test reward (avg +/- std): (497.45 +/- 25.37217964621881) - Num episodes: 100