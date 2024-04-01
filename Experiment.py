import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
import numpy as np
from Agent2 import train
from scipy.signal import savgol_filter
import sys

env = gym.make('CartPole-v1')#, render_mode="human")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_episodes = 500; n_repetitions = 6; 

#Optimal found arguments are listed here 

network_sizes = [256, 256, 256]
lr = 0.5e-4; gamma = 0.99;
eps_start = 0.7; eps_end = 0.05; eps_decay = 700; temp = 1


def dqn(policy="egreedy", er_enabled=True, tn_enabled=True, network_sizes = network_sizes,
    num_episodes = num_episodes, n_repetitions = n_repetitions, lr = lr, gamma = gamma, eps_start = eps_start, eps_end = eps_end, eps_decay = eps_decay, temp = temp,
    save_txt = False, save_plot = False):
    """Runs the network for n_repetitions and returns the average award per episode in the form of both a .pdf plot and .txt file. Further settings
    are included to fit the amount of episodes and repetitions, learning rate, decay rate, epsilon values and the softmax temperature.
        Arguments as follows:
    policy: either 'egreedy' or 'softmax';
    er_enabled: boolean;
    tn_enabled: boolean"""
    network_sizes = [256, 256, 256]
    reward_curve = np.zeros([num_episodes, n_repetitions])
    for n in range(0, n_repetitions):
        print(f"Current at repetition {n}")
        reward_curve[:, n] = train(
        env=env,
        device=device,
        num_episodes=num_episodes,
        buffer_depth=10000,
        batch_size=128,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        tau=0.005,
        lr=lr,
        policy=policy,
        temp=temp,
        network_sizes = network_sizes,
        er_enabled = er_enabled,
        tn_enabled = tn_enabled
        )
    epss = epsilon_start; epse = epsilon_end; epsd = epsilon_decay;
    reward_curve_norm = np.mean(reward_curve, axis=1)
    if save_txt == True:
        np.savetxt(f"reward_curves.txt", reward_curve)
    plt.figure()
    plt.title(f"Average reward curve over {n_repetitions} runs with {policy} policy")
    plt.xlabel("episode"); plt.ylabel("rewards")
    #plt.xticks(np.linspace(0, num_episodes, reward_eval_count+1))
    plt.plot(np.linspace(0, num_episodes, num_episodes+1)[:-1], reward_curve_norm, label = f"er_enabled = {er_enabled}, tr_enabled = {tr_enabled}")
    plt.legend()
    if save_plot == True:
        plt.savefig(f"reward_curve.pdf")
    plt.show()

def main():


    args = sys.argv
    globals()[args[1]](*args[2:]) #Can call function from command line and supply arguments

if __name__ == "__main__":
    main()
