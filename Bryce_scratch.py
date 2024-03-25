import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy   #There is policy Rl and value RL, this is gonna be policy RL.
from rl.memory import SequentialMemory




env = gym.make('CartPole-v1', render_mode="human")
observation, info = env.reset()
states = env.observation_space.shape[0]
print("states{}".format(states))
actions = env.action_space.n

test_episodes = False
if test_episodes:
    episodes = 1
    for episode in range(1, 1 + episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = random.choice([0,1])
            observation, reward, done, truncated, info = env.step(action)
            score += reward
            if done:
                print("Episode {}\tScore: {}".format(episode,score))


def build_model(states, actions):
    '''Give states at the top and get actions at the bottom.'''
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))   #Flate node with 4 different states
    model.add(Dense(24, activation = "relu")) #Linear layer with 24 nodes
    model.add(Dense(24, activation = "relu"))
    model.add(Dense(actions, activation = "linear"))  #Actions
    return model

model = build_model(states, actions)

#Display summary?
summary = False
if summary:
    model.summary()

def build_agent(model,actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 10000, window_length = 1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy,
                   nb_actions = actions, nb_steps_warmup = 10,
                   target_model_update = 1e-2)
    return dqn

#Instantiating DQN model, by using the build_model function passing through our model as well as our actions
dqn = build_agent(model, actions)

#Compiling the DQN model passing through the Adam optimiser.
# The learning rate is 1e-3 and the metric is 'mean absolute error'.
dqn.compile(Adam(lr = 1e-3), metrics = ['mae'])

#Then use fit function to start training.
#Pass through entire environment, the number of steps we want to take
dqn.fit(env, nb_steps = 50000, visualize = False, verbose = 1)

scores = dqn.test(env, nb_episodes = 5, visualize = True)
print(np.mean(scores.history["episode_reward"]))

#dqn.save_weights('dqn_weights', overwrite = True)


# class Model(nn.Module):
#
#     def __init__(self, in_features = 4, h1 = 8, h2 = 8, out_features = 2):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, h1)
#         self.fc2 = nn.Linear(h1, h2)
#         self.out = nn.Linear(h2, out_features)   #Oke dus deze heeft nu dus 3 lagen, moet nog ff kijken hoe je n lagen maakt
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.out(x)
#
#         return x
#
# #Create instance of model:
# torch.manual_seed(41)
# model = Model()
#

#
# env = gym.make('CartPole-v1', render_mode="human")
# observation, info = env.reset()
#
# budget = 10000
# while budget != 0:
#     action = env.action_space.sample()
#
#     observation, reward, done, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         obervation, info = env.reset()
#         print(observation, reward, terminated, truncated, info)
#
#     env.render()
#
#     budget -= 1
