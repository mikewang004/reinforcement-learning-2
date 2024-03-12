class Agent:
    def __init__(self, policy, n_actions, gamma, device):
        self.policy = policy
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.t_step = 0 #hold current episode timestep

    def select_action(self, s, policy=None, epsilon=None, temp=None):


