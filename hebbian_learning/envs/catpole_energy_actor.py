import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', type=bool, default=True,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

writer = SummaryWriter()
# summary_iter = 0
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def rho(s):
    return torch.clamp(s,0.,1.)
    # return torch.clamp(s,-1.,1.)
    # return 2 * F.sigmoid(s) - 1
    # return F.sigmoid(s)

def param_rho(s):
    # return 2 * F.sigmoid(s) - 1
    return torch.clamp(s,-1.,1.)

# MLP Model. Might want to do a vanilla RNN later.
# epsilon: LTD and LTP rate.
# alpha: learning rate of parameter optimization
# eta: weighted average term for energy. Higher term means more of previous energy retained
# delta: multipier for effect of observed reward on parameter optimization
# gamma: Baseline energy added to reward term in parameter optimization.
class Equil_Prop_Policy(nn.Module):
    def __init__(self, input_size, output_size, alpha):
        super(Equil_Prop_Policy, self).__init__()

        # self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["hidden_sizes"] = 128
        # self.hyperparameters["epsilon"] = epsilon
        self.hyperparameters["alpha"] = alpha
        # self.hyperparameters["eta"] = eta
        # self.hyperparameters["delta"] = delta
        # self.hyperparameters["gamma"] = gamma
        
        self.input = torch.zeros([input_size], dtype=torch.float32)
        self.hidden = torch.zeros([self.hyperparameters["hidden_sizes"]], dtype=torch.float32)
        self.output = torch.zeros([output_size], dtype=torch.float32)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        
        self.biases = [torch.zeros(t.shape, requires_grad=True) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t.shape[0], post_t.shape[0]), requires_grad=True)) for pre_t, post_t in zip(self.units[:-1],self.units[1:]) ]
        self.params = self.biases + self.weights

        self.saved_energy = []
        self.model_optimizer = optim.Adam(self.params, lr=alpha)

    # def unit_coorelations(self):
    #     weight_coorelations = [torch.matmul(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
    #     bias_coorelations = [rho(t) for t in self.units]
    #     return weight_coorelations, bias_coorelations

    def energy(self):
        # squared_norm = torch.sum(torch.stack([torch.sum(layer**2) for layer in self.units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(self.units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.dot(torch.matmul(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1)).view(-1), W.view(-1)))
                                     for pre_t,W,post_t in zip(self.units[:-1],self.weights,self.units[1:])]))
        return linear_terms + quadratic_terms

    # Coverges network towards fixed point.
    def forward(self, input):
        # self.input = torch.from_numpy(input).float()
        self.input = input
        self.hidden = torch.mm(self.input.view(1, -1), self.weights[0]).view(-1) + self.biases[1]
        self.hidden = rho(self.hidden)
        self.output = torch.mm(self.hidden.view(1, -1), self.weights[1]).view(-1) + self.biases[2]
        self.units = [self.input, self.hidden, self.output]
        return self.output

    # Does contrastive parameter optimization.
    # Input log likelyhood of action actually taken?
    def optimize(self, losses):
        # for (weight_coorelation, bias_coorelation), loss in zip(self.saved_coorelations, losses):
        #     self.weights = [layer + self.hyperparameters["alpha"] * (-loss * coorelation) for layer, coorelation in zip(self.weights, weight_coorelation)]
        #     self.biases = [layer + 0.1 * self.hyperparameters["alpha"] * (-loss * coorelation) for layer, coorelation in zip(self.biases, bias_coorelation)]
        losses = [loss.detach() for loss in losses]
        energy_loss = [-energy * loss for energy, loss in zip(self.saved_energy, losses)]
        sumed_loss = torch.sum(torch.stack(energy_loss))
        self.model_optimizer.zero_grad()
        sumed_loss.backward(retain_graph=False)
        self.model_optimizer.step()

class Value_Network(nn.Module):
    def __init__(self):
        super(Value_Network, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        # self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # action_scores = self.action_head(x)
        state_values = self.value_head(x)
        # return F.softmax(action_scores, dim=-1), state_values
        return state_values


val_model = Value_Network()
policy_model = Equil_Prop_Policy(4, 2, 0.003)
optimizer = optim.Adam(val_model.parameters(), lr=0.03)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    # probs, state_value = model(state)
    state_value = val_model(state)
    probs = F.softmax(policy_model(state), dim=-1)
    m = Categorical(probs)
    action = m.sample()
    val_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = val_model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in val_model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        # policy_losses.append(-log_prob * reward)
        # policy_losses.append(-reward)
        policy_losses.append(reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    # Critic
    optimizer.zero_grad()
    # loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss = torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    # Actor
    policy_model.optimize(policy_losses)
    del val_model.rewards[:]
    del val_model.saved_actions[:]
    del policy_model.saved_energy[:]


def main():
    running_reward = 10
    summary_iter = 0
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            val_model.rewards.append(reward)
            policy_model.saved_energy.append(policy_model.energy())
            writer.add_scalar('data/percent_0', F.softmax(policy_model.output, dim=-1)[0], summary_iter)
            writer.add_scalar('data/percent_1', F.softmax(policy_model.output, dim=-1)[1], summary_iter)
            summary_iter += 1
            if done:
                break
        writer.add_scalar('data/episode_reward', t, i_episode)
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()