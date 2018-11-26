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

from hebbian_learning.models.equilibrium_propagation_reward_policy import Equilibrium_Propagation_Reward_Policy_Network

parser = argparse.ArgumentParser(description='PyTorch RL Example')
parser.add_argument('--epsilon', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--delta', type=float, default=10)
parser.add_argument('--gamma', type=float, default=10)
parser.add_argument('--n_iterations', type=int, default=5)
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--render', type=bool, default=True)
parser.add_argument('--log-interval', type=int, default=1)
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

def main():
    model = Equilibrium_Propagation_Reward_Policy_Network(reduce(lambda x, y: x*y, env.observation_space.shape), 
                                                          env.action_space.n, args.epsilon, args.alpha, 
                                                          args.eta, args.delta, args.gamma)
    running_reward = 15
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            # pre_energy = model.__energy()
            action_neurons = model.forward(np.reshape(state, -1), args.n_iterations)
            action = Categorical(F.softmax(action_neurons))
            print(F.softmax(action_neurons))
            state, reward, done, _ = env.step(action.sample().item())
            if args.render:
                env.render()
            model.optimize(reward)
            if done:
                model.reset_network()
                break
        running_reward = running_reward * 0.99 + t * 0.01
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(
                i_episode, t, running_reward))
            # print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}\tEnergy: {:.2f} | {:.2f}'.format(
                # i_episode, t, running_reward))
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break


if __name__ == '__main__':
    main()