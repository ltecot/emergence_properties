import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random

class Qt_Opt_Equil_Prop(object):
    def __init__(self, network, num_states, num_actions, target_replace_period, epsilon, gamma):
        self.eval_net = network
        self.target_net = network.copy()
        self.learn_step_counter = 0  # for target updating
        self.num_states = num_states
        self.num_actions = num_actions
        self.target_replace_period = target_replace_period
        self.eps = epsilon
        self.gamma = gamma

    def choose_action(self, x):
        state = torch.from_numpy(x).float().view(-1)
        if np.random.uniform() < self.eps:
            i0 = torch.cat((state.view(-1), torch.tensor([0]).float()))
            v0 = self.eval_net.forward(i0)
            i1 = torch.cat((state.view(-1), torch.tensor([1]).float()))
            v1 = self.eval_net.forward(i1)
            action = 1 if v1 > v0 else 0
            # input = i1 if v1 > v0 else i0
        else:
            action = 1 if np.random.uniform() > 0.5 else 0
            # input = None
        return action # , input

    def target_max(self, s_):
        s_ = torch.tensor(s_).float()
        v0 = self.target_net.forward(torch.cat((s_, torch.tensor([0]).float())))
        v1 = self.target_net.forward(torch.cat((s_, torch.tensor([1]).float())))
        return torch.max(v0, v1)

    def learn(self, s, a, r, done, s_):
        # if self.memory_counter > self.memory_capacity:
            # target parameter update
        if self.learn_step_counter % self.target_replace_period == 0:
            self.target_net.copy_(self.eval_net)
        self.learn_step_counter += 1
        # q_eval = self.eval_net.forward(torch.cat((s.float(), a.float())))
        q_next = self.target_max(s_).detach()     # detach from graph, don't backpropagate
        q_target = r + self.gamma * (1 - done) * q_next   # shape (batch, 1)
        _, _, cost = self.eval_net.optimize(torch.cat((torch.tensor(s).float(), torch.tensor([a]).float())), q_target)
        return cost