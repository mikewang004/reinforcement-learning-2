import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

class ReplayBuffer:
    def __init__(self, buffer_depth):
        self.Step = namedtuple('Step', ('state', 'action', 'next_state', 'reward', 'terminated'))
        self.memory = deque([], maxlen=buffer_depth)

    def push(self, *args):
        self.memory.append(self.Step(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

def select_action(state, steps_done, eps_start, eps_end, eps_decay, env, policy_network, device, policy, temp):
    if policy == "egreedy":
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_network(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    elif policy == "softmax":
        x = policy_network(state)[0].cpu().detach().numpy()
        x = np.array(x) / temp  # scale by temperature
        z = np.array(x) - max(x)  # subtract max to prevent overflow of softmax
        distr = np.exp(z) / np.sum(np.exp(z))  # compute softmax

        selected_action = np.random.choice([0, 1], 1, p=distr)[0]
        return torch.tensor([[selected_action]], device=device, dtype=torch.long)

    else:
        raise KeyError("Choose either 'egreedy' or 'softmax'")

def train_model(memory, policy_network, target_network, optimizer, device, batch_size, gamma, temp):
    if len(memory) < batch_size:
        return

    steps = memory.sample(batch_size)
    batch = memory.Step(*zip(*steps))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_network(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()

def train(env, device, num_episodes, buffer_depth, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, policy):
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    policy_network = QNetwork(n_observations, n_actions).to(device)
    target_network = QNetwork(n_observations, n_actions).to(device)
    target_network.load_state_dict(policy_network.state_dict())

    optimizer = optim.AdamW(policy_network.parameters(), lr=lr, amsgrad=True)
    memory = ReplayBuffer(buffer_depth)

    steps_done = 0
    episode_lengths = []

    for i in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state, steps_done, eps_start, eps_end, eps_decay, env, policy_network, device, policy, temp)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward, done)
            state = next_state

            train_model(memory, policy_network, target_network, optimizer, device, batch_size, gamma)

            target_network_state_dict = target_network.state_dict()
            policy_network_state_dict = policy_network.state_dict()
            for key in policy_network_state_dict:
                target_network_state_dict[key] = policy_network_state_dict[key] * tau + target_network_state_dict[key] * (1 - tau)
            target_network.load_state_dict(target_network_state_dict)

            if done:
                episode_lengths.append(t + 1)
                break

    print('Complete')

    # Plot episode lengths
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.title('Training')
    plt.show()

def main():
    env = gym.make('CartPole-v1', render_mode="human")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(
        env=env,
        device=device,
        num_episodes=20,
        buffer_depth=10000,
        batch_size=128,
        gamma=0.99,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=1000,
        tau=0.005,
        lr=1e-4,
        policy="softmax",
        temp=0.1
    )

if __name__ == "__main__":
    main()
