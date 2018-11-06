import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random

# Hyper Parameters
env = gym.make('CartPole-v0')
LR = 0.001
GAMMA = 0.99  
EPSILON = 0.95
# EPSILON_INIT = 0.1
# EPSILON_FIN = 0.99 
EPSILON_RATE = 100000
BATCH_SIZE = 256
TARGET_REPLACE_ITER = 100 
MEMORY_CAPACITY = 256


# env = gym.make('MountainCar-v0')
# LR = 0.01
# GAMMA = 0.99  
# EPSILON_INIT = 0.1
# EPSILON_FIN = 0.99 
# EPSILON_RATE = 100000
# BATCH_SIZE = 256
# TARGET_REPLACE_ITER = 100 
# MEMORY_CAPACITY = 256

# env = gym.make('Acrobot-v1')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Pendulum-v0')

gym.spaces.prng.seed(1337)
env = env.unwrapped
N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
N_STATES = 1

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 4)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(4, 1)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, states, actions):
        x = self.fc1(states + actions)
        x = F.relu(x)
        value = self.out(x)
        return value
        # return F.softmax(actions_value)

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.eps = EPSILON_INIT
        # self.time_count = 1

    def 

    def choose_action(self, x):
        state = torch.from_numpy(x).float().unsqueeze(0)
        # probs = self.eval_net.forward(state)
        # eps_progress = self.time_count / EPSILON_RATE if self.time_count / EPSILON_RATE < 1 else 1
        # self.eps = EPSILON_INIT * (1 - eps_progress) + EPSILON_FIN * eps_progress
        # self.time_count += 1
        if np.random.uniform() < EPSILON:
            v0 = self.eval_net.forward(state, [0])
            v1 = self.eval_net.forward(state, [1])
            action = 1 if v1 > v0 else 0
            # m = Categorical(probs)
            # action = m.sample()
            # action = action.item()
        else:
            action = env.action_space.sample()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # Run optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

running_reward = 0
# training_done = False
for i_episode in range(100000):
    # if training_done:
    #     break
    s = env.reset()
    ep_r = 0
    for t in range(100000):
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        dqn.store_transition(s, a, r, s_)
        s = s_
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        if done:
            running_reward = running_reward * 0.99 + ep_r * 0.01
            print('Episode {}\treward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_r, running_reward))
            break
            # if running_reward > env.spec.reward_threshold:
            #     print("Solved! Running reward is now {} and "
            #           "the last episode got {} reward!".format(running_reward, ep_r))
            #     training_done = True
            # break
env.close()

