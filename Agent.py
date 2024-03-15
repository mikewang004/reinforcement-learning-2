import torch
from torch import nn
import copy
class Agent():

    def __init__(self, seed, layer_sizes, lr, sync_frequency, rep_buffer_depth, gamma):
        torch.manual_seed(seed)
        self.q_network = self.create_nn(layer_sizes)
        self.target_network = copy.deepcopy(self.q_network)

    def create_nn(self, layers):
        assert len(layer) > 1
        layers = []
        for i in range(len(layers)-1):
            linear = nn.Linear(layers[i], layers[i+1])
            activation = nn.tanh() if i < len(layers)-2 else nn.Identity()
            layers += [linear, activation]
        return nn.Sequential(*layers)