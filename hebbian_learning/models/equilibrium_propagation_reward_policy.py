import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def rho(s):
    # return torch.clamp(s,0.,1.)
    return 2 * F.sigmoid(s) - 1

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
        self.hyperparameters["hidden_sizes"] = 512
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
        
        self.biases = [torch.zeros(t.shape, requires_grad=False) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t.shape[0], post_t.shape[0]), requires_grad=False)) for pre_t, post_t in zip(self.units[:-1],self.units[1:]) ]
        # self.weights = [torch.zeros((pre_t.shape[0], post_t.shape[0]), requires_grad=True) for pre_t, post_t in zip(self.units[:-1],self.units[1:]) ]
        self.params = self.biases + self.weights

        self.energy_optimizer = optim.Adam(self.free_units, lr=epsilon)
        self.model_optimizer = optim.Adam(self.params, lr=alpha)

        # self.running_energy = self.energy()
        # self.running_init_energy = 0
        # self.running_minimized_energy = 0
        self.running_bias_coorelations = [torch.zeros(t.shape) for t in self.biases]
        self.running_weight_coorelations = [torch.zeros(t.shape) for t in self.weights]

    # ENERGY FUNCTION, DENOTED BY E
    def energy(self):
        squared_norm = torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),rho(layer))) for layer in self.units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(self.units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(rho(pre),W),rho(post))) for pre,W,post in zip(self.units[:-1],self.weights,self.units[1:])]))
        return squared_norm + linear_terms + quadratic_terms

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    # def __predict_and_measure(self, ground_truth):
    #     return self.output, self.__energy()

    def __update_running_coorelations(self):
        # self.running_energy = self.running_energy * self.hyperparameters["eta"] + new_energy * (1. - self.hyperparameters["eta"])
        # self.running_energy = self.running_energy + new_energy
        # self.running_init_energy += new_init_energy
        # self.running_minimized_energy = new_minimized_energy
        wc, bc = self.__unit_coorelations()
        self.running_bias_coorelations = [self.hyperparameters["eta"] * t + c for t, c in zip(self.running_bias_coorelations, bc)]
        self.running_weight_coorelations = [self.hyperparameters["eta"] * t + c for t, c in zip(self.running_weight_coorelations, wc)]

    def __unit_coorelations(self):
            weight_coorelations = [torch.mm(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
            bias_coorelations = [rho(t) for t in self.units]
            return weight_coorelations, bias_coorelations

    def reset_network(self):
        # for weight in self.weights:
        #     nn.init.xavier_uniform_(weight)
        # for bias in self.biases:
        #     nn.init.zeros_(bias)
        # self.running_energy = self.energy()
        # self.running_init_energy = 0
        # self.running_minimized_energy = 0
        self.running_bias_coorelations = [torch.zeros(t.shape) for t in self.biases]
        self.running_weight_coorelations = [torch.zeros(t.shape) for t in self.weights]
        # self.hidden = nn.init.normal_(torch.zeros(self.hidden.shape, dtype=torch.float32, requires_grad=True), mean = 0, std=1)
        # self.output = nn.init.normal_(torch.zeros(self.output.shape, dtype=torch.float32, requires_grad=True), mean = 0, std=1)

    # Coverges network towards fixed point.
    def forward(self, input, n_iterations):
        # for layer in self.free_units:
        #     torch.randn(3, 5)
        self.input = torch.from_numpy(input).float()
        # for _ in range(n_iterations):
        #     self.energy_optimizer.zero_grad()
        #     energy = self.energy()
        #     energy.backward()
        #     self.energy_optimizer.step()
        self.hidden = torch.mm(self.input.view(1, -1), self.weights[0]).view(-1) + self.biases[1]
        self.hidden = rho(self.hidden)
        self.output = torch.mm(self.hidden.view(1, -1), self.weights[1]).view(-1) + self.biases[2]
        return self.output

    # Does contrastive parameter optimization. Change to Hebbian later.
    def optimize(self, reward):
        # # self.__update_running_energy(init_energy, self.energy())
        # # target_energy = self.running_energy + self.hyperparameters["gamma"] - self.hyperparameters["delta"] * reward
        # # target_energy = -reward * self.running_energy
        # energy_loss = reward * (self.running_minimized_energy - self.running_init_energy)
        # # target_energy = self.hyperparameters["gamma"] - self.hyperparameters["delta"] * reward
        # # loss = (self.running_energy - target_energy) ** 2
        # self.model_optimizer.zero_grad()
        # # target_energy.backward(retain_graph=True)
        # energy_loss.backward(retain_graph=True)
        # # loss.backward(retain_graph=True)
        # self.model_optimizer.step()
        self.__update_running_coorelations()
        with torch.no_grad():
            self.weights = [layer + self.hyperparameters["alpha"] * 
                            (self.hyperparameters["delta"] * reward * coorelation 
                            - self.hyperparameters["gamma"] * nn.init.normal_(torch.zeros(layer.shape, dtype=torch.float32), mean = 0, std=1))
                            for layer, coorelation in zip(self.weights, self.running_weight_coorelations)]
            self.biases = [layer + self.hyperparameters["alpha"] * 
                          (self.hyperparameters["delta"] * reward * coorelation 
                          - self.hyperparameters["gamma"] * nn.init.normal_(torch.zeros(layer.shape, dtype=torch.float32), mean = 0, std=1))
                        for layer, coorelation in zip(self.biases, self.running_bias_coorelations)]