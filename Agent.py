import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env = gym.make('CartPole-v1', render_mode="human")

is_python = 'inline' in matplotlib.get_backend()
if is_python:
    from IPython import display

plt.ion()

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

step = namedtuple('step', ('state', 'action', 'next_state', 'reward', 'terminated'))

class replaybuffer(object):

    def __init__(self, buffer_depth):
        self.memory = deque([], maxlen = buffer_depth)

    #Save a step
    def push(self, *args):
        self.memory.append(step(*args))

    #Get a random sample
    def sample(self, batch_size):
        """
        A function that samples a specified number of elements from the memory.

        Parameters:
            batch_size (int): The number of elements to sample from the memory.

        Returns:
            list: A list of randomly sampled elements from the memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the length of the memory list.

        Returns:
        int: The length of the memory list.
        """
        return len(self.memory)

#Q-network
class Q_network(nn.Module):
    # make function to create a network
    def __init__(self, n_states, n_actions):
        super(Q_network, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

#HYPERPARAMETERS & plotting
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_network = Q_network(n_observations, n_actions).to(device)
target_network = Q_network(n_observations, n_actions).to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.AdamW(policy_network.parameters(), lr = LR, amsgrad = True)
memory = replaybuffer(10000)

steps_done = 0


#select action function
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_network(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_lengths = []

def plot_lengths(show_result=False):
    plt.figure(1)
    durations = torch.tensor(episode_lengths, dtype=torch.float)
    if show_result:
        plt.title('result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations.numpy())
    # Take 100 episode averages and plot them too
    if len(durations) >= 100:
        means = durations.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.00)
    if is_python:
        if not show_plot:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else: display.display(plt.gcf())


#function to train model
def train_model():
    if len(memory) < BATCH_SIZE:
        return

    steps = memory.sample(BATCH_SIZE)
    batch = step(*zip(*steps))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_network(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()

#Performing training
if torch.cuda.is_available():
    print('CUDA is available')
    num_episodes = 300
else:
    print('CUDA is not available')
    num_episodes = 30

for i in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        #store step in memory
        memory.push(state, action, next_state, reward, done)

        #update state
        state = next_state

        #perform training step
        train_model()

        target_network_state_dict = target_network.state_dict()
        policy_network_state_dict = policy_network.state_dict()
        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key] * TAU + target_network_state_dict[key] * (1-TAU)
        target_network.load_state_dict(target_network_state_dict)

        if done:
            episode_lengths.append(t + 1)
            plot_lengths()
            break

print('Complete')
plot_lengths(show_result=True)
plt.ioff()
plt.show()