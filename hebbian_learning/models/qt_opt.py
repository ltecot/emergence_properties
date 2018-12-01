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
                # self.target_net.load_state_dict(self.eval_net.state_dict())
                self.target_net.copy_(self.eval_net)
            self.learn_step_counter += 1
            sample_index = np.random.choice(self.memory_capacity, self.batch_size )
            b_memory = self.memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :self.num_states])
            b_a = torch.LongTensor(b_memory[:, self.num_states:self.num_states+self.num_actions])
            b_r = torch.FloatTensor(b_memory[:, self.num_states+self.num_actions:self.num_states+self.num_actions+1])
            b_t = torch.FloatTensor(b_memory[:, self.num_states+self.num_actions+1:self.num_states+self.num_actions+2])
            b_s_ = torch.FloatTensor(b_memory[:, -self.num_states:])
            q_eval = self.eval_net(torch.cat((b_s.float(), b_a.float()), dim=1))
            q_next = self.target_max(b_s_).detach()     # detach from graph, don't backpropagate
            q_target = b_r + self.gamma * (1 - b_t) * q_next   # shape (batch, 1)
            self.eval_net.optimize(q_eval, q_target)
            # loss = self.loss_func(q_eval, q_target)
            # Run optimizer
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

# Hyper Parameters
env = gym.make('CartPole-v0')
LR = 0.01
GAMMA = 0.99  
EPSILON = 0.95
BATCH_SIZE = 16
TARGET_REPLACE_ITER = 10
MEMORY_CAPACITY = 256

env.seed(1337)
torch.manual_seed(1337)
env = env.unwrapped
N_ACTIONS = 1 # env.action_space.n
N_STATES = env.observation_space.shape[0]

network = MLP(N_STATES + N_ACTIONS, 1, 64, LR)
rl_model = Qt_Opt(network, N_STATES, N_ACTIONS, MEMORY_CAPACITY, TARGET_REPLACE_ITER, BATCH_SIZE, EPSILON, GAMMA)

running_reward = 20
for i_episode in range(100000):
    s = env.reset()
    ep_r = 0
    for t in range(100000):
        env.render()
        a = rl_model.choose_action(s)
        s_, r, done, info = env.step(a)
        rl_model.store_transition(s, a, r, done, s_)
        s = s_
        ep_r += r
        rl_model.learn()
        if done:
            running_reward = running_reward * 0.99 + ep_r * 0.01
            print('Episode {}\treward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_r, running_reward))
            break
env.close()