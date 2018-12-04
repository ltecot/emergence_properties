import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random

class Qt_Opt(object):
    def __init__(self, network, num_states, num_actions, memory_capacity, batch_size, target_replace_period, epsilon, gamma):
        self.eval_net = network
        self.target_net = network.copy()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((memory_capacity, num_states * 2 + num_actions + 2))     # initialize memory
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.target_replace_period = target_replace_period
        self.eps = epsilon
        self.gamma = gamma

    def choose_action(self, x):
        state = torch.from_numpy(x).float().view(-1)
        if np.random.uniform() < self.eps:
            v0 = self.eval_net.forward(torch.cat((state.view(1, -1), torch.tensor([[0]]).float()), dim=1))
            v1 = self.eval_net.forward(torch.cat((state.view(1, -1), torch.tensor([[1]]).float()), dim=1))
            action = 1 if v1 > v0 else 0
        else:
            action = 1 if np.random.uniform() > 0.5 else 0
        return action

    def store_transition(self, s, a, r, done, s_):
        transition = np.hstack((s, [a, r, done], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def target_max(self, b_s_):
        v0 = self.target_net.forward(torch.cat((b_s_, torch.zeros(b_s_.shape[0], 1).float()), dim=1))
        v1 = self.target_net.forward(torch.cat((b_s_, torch.ones(b_s_.shape[0], 1).float()), dim=1))
        return torch.max(v0, v1)

    def learn(self):
        if self.memory_counter > self.memory_capacity:
            # target parameter update
            if self.learn_step_counter % self.target_replace_period == 0:
                self.target_net.copy_(self.eval_net)
            self.learn_step_counter += 1
            sample_index = np.random.choice(self.memory_capacity, self.batch_size )
            b_memory = self.memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :self.num_states])
            b_a = torch.LongTensor(b_memory[:, self.num_states:self.num_states+self.num_actions])
            b_r = torch.FloatTensor(b_memory[:, self.num_states+self.num_actions:self.num_states+self.num_actions+1])
            b_t = torch.FloatTensor(b_memory[:, self.num_states+self.num_actions+1:self.num_states+self.num_actions+2])
            b_s_ = torch.FloatTensor(b_memory[:, -self.num_states:])
            q_eval = self.eval_net.forward(torch.cat((b_s.float(), b_a.float()), dim=1)).detach()
            q_next = self.target_max(b_s_).detach()     # detach from graph, don't backpropagate
            q_target = b_r + self.gamma * (1 - b_t) * q_next   # shape (batch, 1)
            self.eval_net.optimize(q_eval, q_target)