import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
        weight_coorelations = [torch.mm(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
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
                       (0#self.hyperparameters["delta"] * reward * coorelation  # Reward Update
                      # - self.hyperparameters["delta"] * self.step_count * self.average_reward * average  # Negative Reward Update
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
        # self.biases = [layer * 0.995 for layer in self.biases]
        # self.weights = [layer * 0.995 for layer in self.weights]