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

def experiment_epsilon(epsilon_start, epsilon_end, epsilon_decay, env, device, num_episodes, network_sizes, n_repetitions = 20):
    repetition_array = np.zeros(n_repetitions)
    #reward_curve_repetitions = np.zeros([n_repetitions, reward_eval_count, len(epsilon_decay) * len(epsilon_end) * len(epsilon_start)]) #Store reward curves here
    reward_curve_repetitions = np.zeros([num_episodes, len(epsilon_decay) * len(epsilon_end) * len(epsilon_start)])
    i = 0
    for epsd in epsilon_decay:
        for epse in epsilon_end:
            for epss in epsilon_start:
                for n in range(0, n_repetitions):
                    print(f"Current at repetition {n}")
                    reward_curve_single_setting = np.zeros([n_repetitions, num_episodes])
                    reward_curve_single_setting[n, :] = train(
                    env=env,
                    device=device,
                    num_episodes=num_episodes,
                    buffer_depth=10000,
                    batch_size=128,
                    gamma=0.99,
                    eps_start=epss,
                    eps_end=epse,
                    eps_decay=epsd,
                    tau=0.005,
                    lr=1e-4,
                    policy="egreedy",
                    temp=0.1,
                    network_sizes = network_sizes,
                    er_enabled = True,
                    tn_enabled = True
                    )
                reward_curve_repetitions[:, i] = np.mean(reward_curve_single_setting, axis = 0)
                i = i + 1
                print(f"Current at iteration {i} of {i*len(epsilon_decay) * len(epsilon_end) * len(epsilon_start)}, with epsilon start = {epss}, epsilon end = {epse}, epsilon decay = {epsd}")
    #Take average of n_repetitions
    print(reward_curve_repetitions.shape)
    reward_curve = reward_curve_repetitions #Note episodes at axis 0; num iterations at axis 1
    #Plot reward curves 
    i = 0
    np.savetxt("data/epsilon_reward_curves-2.txt", reward_curve)
    for epsd in epsilon_decay:
        plt.figure()
        plt.title("Comparison of various epsilon values")
        plt.xlabel("episode"); plt.ylabel("rewards")
        #plt.xticks(np.linspace(0, num_episodes, reward_eval_count+1))
        for epse in epsilon_end:
            for epss in epsilon_start:
                plt.plot(np.linspace(0, num_episodes, num_episodes+1)[:-1], reward_curve[:, i], label = f"eps_start = {epss}, eps_end = {epse}, eps_decay = {epsd}")
                i = i + 1
        plt.legend(loc='upper left', prop={'size': 6})
        plt.savefig(f"plots/learning_curve_epsd_{epsd}_num_eps_{num_episodes}_n_reps_{n_repetitions}.pdf")
    


def single_plot(epsilon_start, epsilon_end, epsilon_decay, env, device, num_episodes, network_sizes, n_repetitions = 20):
    reward_curve = np.zeros([num_episodes, n_repetitions])
    for n in range(0, n_repetitions):
        print(f"Current at repetition {n}")
        reward_curve[:, n] = train(
        env=env,
        device=device,
        num_episodes=num_episodes,
        buffer_depth=10000,
        batch_size=128,
        gamma=0.99,
        eps_start=epsilon_start,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay,
        tau=0.005,
        lr=1e-4,
        policy="egreedy",
        temp=0.1,
        network_sizes = network_sizes,
        er_enabled = True,
        tn_enabled = True
        )
    epss = epsilon_start; epse = epsilon_end; epsd = epsilon_decay;
    reward_curve_norm = np.mean(reward_curve, axis=1)
    np.savetxt(f"data/reward_curves_epsilons{epss}_{epse}_{epsd}_num_eps_{num_episodes}_n_reps_{n_repetitions}-relu-64-128-64.txt", reward_curve)
    plt.figure()
    plt.title("Average reward curve of epsilon-greedy learning method")
    plt.xlabel("episode"); plt.ylabel("rewards")
    #plt.xticks(np.linspace(0, num_episodes, reward_eval_count+1))
    plt.plot(np.linspace(0, num_episodes, num_episodes+1)[:-1], reward_curve_norm, label = f"eps_start = {epss}, eps_end = {epse}, eps_decay = {epsd}")
    plt.legend()
    plt.savefig(f"plots/learning_curve_epsilons{epss}_{epse}_{epsd}_num_eps_{num_episodes}_n_reps_{n_repetitions}-relu-64-128-64.pdf")
    # Plot average of all curves    

def main():


    env = gym.make('CartPole-v1')#, render_mode="human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_episodes = 500; n_repetitions = 6;
    epsilon_start = [0.8, 0.7]; epsilon_end = [0.05, 0.1]; #epsilon_decay = [100, 500, 1000]
    epsilon_decay = [500, 700]
    network_sizes = [128,256,128]
    #experiment_epsilon(epsilon_start, epsilon_end, epsilon_decay, env, device, num_episodes, network_sizes, n_repetitions = n_repetitions)
    single_plot(epsilon_start = 0.7, epsilon_end = 0.05, epsilon_decay = 700, env=env, device=device, num_episodes= num_episodes, network_sizes= network_sizes, n_repetitions= n_repetitions)


if __name__ == "__main__":
    main()