import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def rho(s):
    return torch.clamp(s,0.,1.)

# MLP Model. Might want to do a vanilla RNN later.
# epsilon: learning rate of energy optimization
# alpha: learning rate of parameter optimization
# eta: weighted average term for energy. Higher term means more of previous energy retained
# delta: multipier for effect of observed reward on parameter optimization
# gamma: Baseline energy added to reward term in parameter optimization.
class Equilibrium_Propagation_Reward_Policy_Network(nn.Module):
    def __init__(self, input_size, output_size, epsilon, alpha, eta, delta, gamma, name="equil_prop_reward_policy"):
        super(Equilibrium_Propagation_Reward_Policy_Network, self).__init__()

        self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["hidden_sizes"] = 128
        self.hyperparameters["epsilon"] = epsilon
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["eta"] = eta
        self.hyperparameters["delta"] = delta
        self.hyperparameters["gamma"] = gamma
        
        self.input = torch.zeros([input_size], dtype=torch.float32)
        self.hidden = torch.zeros([self.hyperparameters["hidden_sizes"]], dtype=torch.float32, requires_grad=True)
        self.output = torch.zeros([output_size], dtype=torch.float32, requires_grad=True)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        
        self.biases = [torch.zeros(t.shape, requires_grad=True) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t.shape[0], post_t.shape[0]), requires_grad=True)) for pre_t, post_t in zip(self.units[:-1],self.units[1:]) ]
        self.params = self.biases + self.weights

        self.energy_optimizer = optim.Adam(self.free_units, lr=epsilon)
        self.model_optimizer = optim.Adam(self.params, lr=alpha)

        self.running_energy = self.__energy()

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self):
        squared_norm = torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),rho(layer))) for layer in self.units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(self.units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(rho(pre),W),rho(post))) for pre,W,post in zip(self.units[:-1],self.weights,self.units[1:])]))
        return squared_norm + linear_terms + quadratic_terms

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    # def __predict_and_measure(self, ground_truth):
    #     return self.output, self.__energy()

    def __update_running_energy(self, new_energy):
        self.running_energy = self.running_energy * self.hyperparameters["eta"] + new_energy * (1. - self.hyperparameters["eta"])

    def reset_network(self):
        # for weight in self.weights:
        #     nn.init.xavier_uniform_(weight)
        # for bias in self.biases:
        #     nn.init.zeros_(bias)
        self.running_energy = self.__energy()

    # Coverges network towards fixed point.
    def forward(self, input, n_iterations):
        self.input = input
        for _ in range(n_iterations):
            self.energy_optimizer.zero_grad()
            energy = self.__energy()
            energy.backward()
            self.energy_optimizer.step()
        return self.output

    # Does contrastive parameter optimization. Change to Hebbian later.
    def optimize(self, reward):
        self.__update_running_energy(self.__energy())
        target_energy = self.running_energy + self.hyperparameters["gamma"] - self.hyperparameters["delta"] * reward
        self.model_optimizer.zero_grad()
        target_energy.backward(retain_graph=True)
        self.model_optimizer.step()