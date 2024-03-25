import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from itertools import count
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
env = gym.make('CartPole-v1')
# print("Actions: {}".format(env.action_space.n))

class ActorCritic(nn.Module):
    #def __init__(self, in_features = 4, h1 = 8, h2 = 8, out_features = 2, gamma = 0.99, n_episodes = 10000, alpha = 0.05):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)   #Oke dus deze heeft nu dus 3 lagen, moet nog ff kijken hoe je n lagen maakt
        self.saved_actions = []
        self.rewards = []
        #self.gamma = gamma
        #self.n_episodes = n_episodes
        #self.alpha = alpha

    def forward(self, x):
        # print("x", x)
        # print("x", self.fc1(x))
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.actor(x), dim = -1)
        state_values = self.critic(x)
        return action_prob, state_values

def select_action(state):
    '''Action agent'''
    state = torch.from_numpy(state[0]).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode(gamma, reward):
    #Calculate the losses and perform backpropagation
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for i in model.rewards[::-1]:
        R = reward + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)
eps = np.finfo(np.float32).eps.item()

def train(n_episodes, alpha = 0.05, gamma = 0.99, episode_length):

    running_reward = 10
    for i_episode in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        for t in range(1,episode_length):
            action = select_action(state)

            observation, reward, done, truncated, info = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = alpha * ep_reward + (1 - alpha) * running_reward
        finish_episode(gamma, reward)
        if i_episode % 10 == 0:
            print('Episode: {}, \tReward: {}'.format(i_episode, running_reward))
        if running_reward > env.spec.reward_threshold:
            print('Episode finished after {}'.format(i_episode))
            break



train(n_episodes = 1000, alpha = 0.05, gamma = 0.99, episode_length = 50000)


