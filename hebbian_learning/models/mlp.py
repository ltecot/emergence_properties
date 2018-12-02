import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random

class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, learning_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc1.weight.data.normal_(0, 0.01)   # initialization
        self.out = nn.Linear(hidden, n_out)
        self.out.weight.data.normal_(0, 0.01)   # initialization

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.n_in = n_in
        self.n_out = n_out
        self.hidden = hidden
        self.learning_rate = learning_rate

    def forward(self, input):
        # x = self.fc1(torch.cat((states.float(), actions.float()), dim=1))
        x = self.fc1(input)
        x = F.relu(x)
        y = self.out(x)
        return y
        # return F.softmax(actions_value)

    def optimize(self, pred, target):
        loss = self.loss_func(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def copy(self):
        new_net = MLP(self.n_in, self.n_out, self.hidden, self.learning_rate)
        new_net.load_state_dict(self.state_dict())
        return new_net

    # In place copy
    def copy_(self, source_net):
        self.load_state_dict(source_net.state_dict())
