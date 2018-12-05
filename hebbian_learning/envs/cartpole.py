import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from hebbian_learning.models.equilibrium_propagation_value import Equilibrium_Propagation_Value_Network
from hebbian_learning.models.qt_opt_equil_prop import Qt_Opt_Equil_Prop
from hebbian_learning.models.mlp import MLP
from hebbian_learning.models.qt_opt import Qt_Opt

parser = argparse.ArgumentParser(description='PyTorch RL Example')
parser.add_argument('--equil_prop', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--render', type=bool, default=True)
parser.add_argument('--log-interval', type=int, default=1)
# Equil Prop
parser.add_argument('--energy_learn_rate', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--epsilon', type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--target_replace_period', type=int, default=10)
# parser.add_argument('--memory_capacity', type=int, default=256)
parser.add_argument('--num_hidden', type=int, default=64)
parser.add_argument('--n_iterations', type=int, default=3)
parser.add_argument('--n_iterations_neg', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.5)
# MLP
# parser.add_argument('--learning_rate', type=float, default=0.01)
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--epsilon', type=float, default=0.95)
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--target_replace_period', type=int, default=10)
# parser.add_argument('--memory_capacity', type=int, default=256)
# parser.add_argument('--num_hidden', type=int, default=64)
args = parser.parse_args()
# args.beta = -np.log(1-args.beta)

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')
# env = env.unwrapped
env.seed(args.seed)
torch.manual_seed(args.seed)
writer = SummaryWriter()
N_ACTIONS = 1 # env.action_space.n
N_STATES = env.observation_space.shape[0]

def main():
    if args.equil_prop:
        network = Equilibrium_Propagation_Value_Network(N_STATES + N_ACTIONS, 1, args.num_hidden,
                                                       args.energy_learn_rate, args.learning_rate, args.n_iterations, 
                                                       args.n_iterations_neg, args.beta)
        rl_model = Qt_Opt_Equil_Prop(network, N_STATES, N_ACTIONS, args.target_replace_period, args.epsilon, args.gamma)
    else:
        network = MLP(N_STATES + N_ACTIONS, 1, args.num_hidden, args.learning_rate)
        rl_model = Qt_Opt(network, N_STATES, N_ACTIONS, args.memory_capacity, 
                          args.batch_size, args.target_replace_period, args.epsilon, args.gamma)
    running_reward = 20
    for i_episode in range(100000):
        s = env.reset()
        ep_r = 0
        for t in range(100000):
            if args.render:
                env.render()
            a = rl_model.choose_action(s)
            s_, r, done, info = env.step(a)
            if not args.equil_prop:
                rl_model.store_transition(s, a, r, done, s_)
                rl_model.learn()
            else:
                rl_model.learn(s, a, r, done, s_)
            s = s_
            ep_r += r
            if done:
                writer.add_scalar('data/episode_reward', t, i_episode)
                running_reward = running_reward * 0.99 + ep_r * 0.01
                print('Episode {}\treward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_r, running_reward))
                break
    env.close()

if __name__ == '__main__':
    main()