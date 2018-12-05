import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import random
import argparse
from itertools import count
from collections import namedtuple
from functools import reduce
from tensorboardX import SummaryWriter

def rho(s):
    return torch.clamp(s,0.,1.)

def param_rho(s):
    return torch.clamp(s,-1.,1.)

# MLP Model. Might want to do a vanilla RNN later.
# epsilon: LTD and LTP rate.
# alpha: learning rate of parameter optimization
# eta: weighted average term for energy. Higher term means more of previous energy retained
# delta: multipier for effect of observed reward on parameter optimization
# gamma: Baseline energy added to reward term in parameter optimization.
class Equilibrium_Propagation_Reward_Policy_Network(nn.Module):
    def __init__(self, input_size, output_size, epsilon, alpha, eta, delta, gamma, name="equil_prop_reward_policy"):
        super(Equilibrium_Propagation_Reward_Policy_Network, self).__init__()

        self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["hidden_sizes"] = 1024
        self.hyperparameters["epsilon"] = epsilon
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["eta"] = eta
        self.hyperparameters["delta"] = delta
        self.hyperparameters["gamma"] = gamma
        
        self.input = torch.zeros([input_size], dtype=torch.float32)
        self.hidden = torch.zeros([self.hyperparameters["hidden_sizes"]], dtype=torch.float32)
        self.output = torch.zeros([output_size], dtype=torch.float32)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        
        self.biases = [torch.zeros(t.shape) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t.shape[0], post_t.shape[0]))) for pre_t, post_t in zip(self.units[:-1],self.units[1:]) ]
        self.params = self.biases + self.weights

        self.running_bias_coorelations = [torch.zeros(t.shape) for t in self.biases]
        self.running_weight_coorelations = [torch.zeros(t.shape) for t in self.weights]
        self.average_weight_coorelations, self.average_bias_coorelations = self.__unit_coorelations()
        self.average_reward = 0
        self.step_count = 0

    def __update_running_coorelations(self):
        wc, bc = self.__unit_coorelations()
        self.running_bias_coorelations = [self.hyperparameters["eta"] * t + c for t, c in zip(self.running_bias_coorelations, bc)]
        self.running_weight_coorelations = [self.hyperparameters["eta"] * t + c for t, c in zip(self.running_weight_coorelations, wc)]
        self.average_bias_coorelations = [self.hyperparameters["eta"] * t + (1 - self.hyperparameters["eta"]) * c 
                                          for t, c in zip(self.average_bias_coorelations, bc)]
        self.average_weight_coorelations = [self.hyperparameters["eta"] * t + (1 - self.hyperparameters["eta"]) * c 
                                          for t, c in zip(self.average_weight_coorelations, wc)]
        self.step_count = self.hyperparameters["eta"] * self.step_count + 1
        return wc, bc

    def __unit_coorelations(self):
        weight_coorelations = [torch.matmul(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
        bias_coorelations = [rho(t) for t in self.units]
        return weight_coorelations, bias_coorelations

    def reset_network(self):
        self.running_bias_coorelations = [torch.zeros(t.shape) for t in self.biases]
        self.running_weight_coorelations = [torch.zeros(t.shape) for t in self.weights]
        self.step_count = 0

    # Coverges network towards fixed point.
    def forward(self, input, n_iterations):
        self.input = torch.from_numpy(input).float()
        self.hidden = torch.mm(self.input.view(1, -1), self.weights[0]).view(-1) + self.biases[1]
        self.hidden = rho(self.hidden)
        self.output = torch.mm(self.hidden.view(1, -1), self.weights[1]).view(-1) + self.biases[2]
        self.units = [self.input, self.hidden, self.output]
        return self.output

    # Does contrastive parameter optimization.
    # Input log likelyhood of action actually taken?
    def optimize(self, reward):
        _, bc = self.__update_running_coorelations()
        self.average_reward = self.average_reward * self.hyperparameters["eta"] + reward * (1 - self.hyperparameters["eta"])
        self.weights = [layer + self.hyperparameters["alpha"] * 
                        (self.hyperparameters["delta"] * reward * coorelation  # Positive Reward Update
                        - self.hyperparameters["delta"] * self.average_reward * average  # Negative Reward Update
                        # - self.hyperparameters["gamma"] * nn.init.normal_(torch.zeros(layer.shape, dtype=torch.float32), mean = 0, std=1)  # Noise
                        ) for layer, coorelation, average in zip(self.weights, self.running_weight_coorelations, self.average_weight_coorelations)]
        self.biases = [layer + self.hyperparameters["alpha"] * 
                       (-self.hyperparameters["delta"] * reward * coorelation  # Reward Update
                       + self.hyperparameters["delta"] * self.step_count * self.average_reward * average  # Negative Reward Update
                    #    - self.hyperparameters["epsilon"] * coorelation  # LTD and LTP
                    #    - self.hyperparameters["gamma"] * nn.init.normal_(torch.zeros(layer.shape, dtype=torch.float32), mean = 0, std=1)  # Noise
                       ) for layer, coorelation, average in zip(self.biases, self.running_bias_coorelations, self.average_bias_coorelations)]
        # Regularization (probably do something smarter here)
        # self.weights = [layer / torch.sum(torch.abs(layer)) for layer in self.weights]
        # self.biases = [layer / torch.sum(torch.abs(layer)) for layer in self.biases]
        # self.weights = [param_rho(layer) for layer in self.weights]
        # self.biases = [param_rho(layer) for layer in self.biases]
        # self.hyperparameters["gamma"] = 0.999 * self.hyperparameters["gamma"]
        # print(self.hyperparameters["gamma"])
        # self.hyperparameters["epsilon"] = 0.999 * self.hyperparameters["epsilon"]
        # print(self.hyperparameters["epsilon"])
        # self.hyperparameters["alpha"] = 0.999 * self.hyperparameters["alpha"]
        # print(self.hyperparameters["alpha"])
        self.biases = [layer - layer / (torch.abs(layer) + 0.000001) * 0.000001 for layer in self.biases]
        self.weights = [layer - layer / (torch.abs(layer) + 0.000001) * 0.000001 for layer in self.weights]



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