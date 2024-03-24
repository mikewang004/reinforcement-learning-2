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

def experiment_epsilon(epsilon_start, epsilon_end, epsilon_decay, env, device, num_episodes, eval_percentage, n_repetitions = 20):
    reward_eval_count = int(eval_percentage * num_episodes)
    repetition_array = np.zeros(n_repetitions)
    reward_curve_repetitions = np.zeros([n_repetitions, reward_eval_count, len(epsilon_decay) * len(epsilon_end) * len(epsilon_start)]) #Store reward curves here
    i = 0
    for epsd in epsilon_decay:
        for epse in epsilon_end:
            for epss in epsilon_start:
                for n in range(0, n_repetitions):
                    reward_curve_repetitions[n, :, i] = train(
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
                    policy="softmax",
                    temp=0.1,
                    reward_eval_count = reward_eval_count
                    )
                i = i + 1
                print(f"Current at iteration {i}")
    #Take average of n_repetitions
    reward_curve = np.mean(reward_curve_repetitions, axis = 0)
    #Plot reward curves 
    i = 0
    for epsd in epsilon_decay:
        plt.figure()
        plt.title("Comparison of various epsilon values")
        plt.xlabel("episode"); plt.ylabel("rewards")
        plt.xticks(np.linspace(0, num_episodes, reward_eval_count+1))
        for epse in epsilon_end:
            for epss in epsilon_start:
                plt.plot(np.linspace(0, num_episodes, reward_eval_count+1)[:-1], reward_curve[:, i], label = f"eps_start = {epss}, eps_end = {epse}, eps_decay = {epsd}")
                i = i + 1
        plt.legend(loc='upper left', prop={'size': 6})
        plt.savefig(f"plots/learning_curve_epsd_{epsd}.pdf")

        
def main():


    env = gym.make('CartPole-v1')#, render_mode="human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_episodes = 300; eval_percentage = 0.05; n_repetitions = 20
    epsilon_start = [0.9, 0.8, 0.7]; epsilon_end = [0.05, 0.1, 0.3]; epsilon_decay = [100, 500, 1000]
    experiment_epsilon(epsilon_start, epsilon_end, epsilon_decay, env, device, num_episodes, eval_percentage, n_repetitions = n_repetitions)

if __name__ == "__main__":
    main()