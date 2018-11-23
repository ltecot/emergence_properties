# import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def rho(s):
    return torch.clamp(s,0.,1.)

# MLP Model. Might want to do a vanilla RNN later.
class Equilibrium_Propagation_Network(nn.Module):
    def __init__(self, input_size, output_size, epsilon, alpha, name="equil_prop_baseline"):
        super(Equilibrium_Propagation_Network, self).__init__()

        self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["hidden_sizes"] = 128
        # self.hyperparameters["batch_size"] = 32
        # self.hyperparameters["learning_rate"] = learning_rate
        
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

        self.model_version = 1  # 1 is gradient-based, 2 is contrastive based.

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self):
        squared_norm = torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),rho(layer))) for layer in self.units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(self.units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(rho(pre),W),rho(post))) for pre,W,post in zip(self.units[:-1],self.weights,self.units[1:])]))
        return squared_norm + linear_terms + quadratic_terms

    # COST FUNCTION, DENOTED BY C
    def __cost(self, ground_truth):
        return torch.sum((self.output - ground_truth) ** 2)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, beta, ground_truth):
        if beta == 0:
            return self.__energy()
        else:
            return self.__energy() + beta * self.__cost(ground_truth)

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __predict_and_measure(self, ground_truth):
        E = self.__energy()
        C = self.__cost(ground_truth)
        # y_softmax = F.log_softmax(self.output, dim=0)
        # return y_softmax, E, C
        return self.output, E, C

    # Coverges network towards fixed point.
    def forward(self, input, n_iterations, beta=0, ground_truth=None):
        if self.model_version == 1:
            self.input = input
            for _ in range(n_iterations):
                self.energy_optimizer.zero_grad()
                # energy = torch.stack(self.__total_energy(beta, ground_truth)).sum()
                energy = self.__total_energy(beta, ground_truth)
                energy.backward()
                self.energy_optimizer.step()
                # self.free_units = [torch.clamp(layer,0.,1.) for layer in self.free_units]
                # for layer in self.free_units:
                    # layer = torch.clamp(layer,0.,1.)
                    # layer.clamp_(0.,1.)
            # return F.log_softmax(self.output, dim=0)
        else:
            self.input = input
            for _ in range(n_iterations):
                self.energy_optimizer.zero_grad()
                # energy = torch.stack(self.__total_energy(beta, ground_truth)).sum()
                energy = self.__total_energy(beta, ground_truth)
                energy.backward()
                self.energy_optimizer.step()

    # Does contrastive parameter optimization. Change to Hebbian later.
    def optimize(self, input, n_iterations_pos, n_iterations_neg, beta, ground_truth):
        if self.model_version == 1:
            self.forward(input, n_iterations_pos, beta=0, ground_truth=None)  # Free Phase
            free_energy = self.__total_energy(0, ground_truth)
            y_pred, mean_free_energy, mean_free_cost = self.__predict_and_measure(ground_truth)
            self.forward(input, n_iterations_neg, beta=beta, ground_truth=ground_truth)  # Constrained Phase
            constrained_energy = self.__total_energy(beta, ground_truth)
            self.model_optimizer.zero_grad()
            # param_loss = torch.stack((constrained_energy - free_energy) / beta).sum()
            param_loss = (constrained_energy - free_energy) # / beta
            param_loss.backward()
            self.model_optimizer.step()
            return y_pred, mean_free_energy, mean_free_cost
        else:
            self.forward(input, n_iterations_pos, beta=0, ground_truth=None)  # Free Phase
            free_energy = self.__total_energy(0, ground_truth)
            y_pred, mean_free_energy, mean_free_cost = self.__predict_and_measure(ground_truth)
            self.forward(input, n_iterations_neg, beta=beta, ground_truth=ground_truth)  # Constrained Phase
            constrained_energy = self.__total_energy(beta, ground_truth)
            self.model_optimizer.zero_grad()
            # param_loss = torch.stack((constrained_energy - free_energy) / beta).sum()
            param_loss = (constrained_energy - free_energy) # / beta
            param_loss.backward()
            self.model_optimizer.step()
            return y_pred, mean_free_energy, mean_free_cost