# import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def rho(s):
    # return torch.clamp(s,0.,1.)
    return torch.sigmoid(s)

# MLP Model. Might want to do a vanilla RNN later.
class Equilibrium_Propagation_Network(nn.Module):
    def __init__(self, input_size, output_size, num_hidden, batch_size, energy_learn_rate, param_learn_rate, num_iterations, num_iterations_neg, beta):
        super(Equilibrium_Propagation_Network, self).__init__()

        self.model_version = 1  # 1 is gradient-based, 2 is contrastive based.
        if self.model_version == 1:
            grad_req = True
        else:
            grad_req = False

        # self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["input_size"] = input_size
        self.hyperparameters["output_size"] = output_size
        self.hyperparameters["batch_size"] = batch_size
        self.hyperparameters["num_hidden"] = num_hidden
        self.hyperparameters["energy_learn_rate"] = energy_learn_rate
        self.hyperparameters["param_learn_rate"] = param_learn_rate
        self.hyperparameters["num_iterations"] = num_iterations
        self.hyperparameters["num_iterations_neg"] = num_iterations_neg
        self.hyperparameters["beta"] = beta
        
        # self.input = torch.zeros([input_size], dtype=torch.float32)
        # self.hidden = torch.zeros([self.hyperparameters["hidden_sizes"]], dtype=torch.float32, requires_grad=grad_req)
        # self.output = torch.zeros([output_size], dtype=torch.float32, requires_grad=grad_req)
        # self.units = [self.input, self.hidden, self.output]
        # self.free_units = [self.hidden, self.output]
        self.input = torch.zeros([batch_size, input_size], dtype=torch.float32)
        self.hidden = torch.zeros([batch_size, num_hidden], dtype=torch.float32, requires_grad=grad_req)
        self.output = torch.zeros([batch_size, output_size], dtype=torch.float32, requires_grad=grad_req)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        
        self.biases = [torch.zeros(t[0].shape, requires_grad=grad_req) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t[0].shape[0], post_t[0].shape[0]), requires_grad=grad_req)) for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
        self.params = self.biases + self.weights

        self.energy_optimizer = optim.Adam(self.free_units, lr=energy_learn_rate)
        self.model_optimizer = optim.Adam(self.params, lr=param_learn_rate)

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, batch_size):
        batch_units = [layer[0:batch_size] for layer in self.units]
        # squared_norm = torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),rho(layer))) for layer in batch_units])) / 2.
        # linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(batch_units,self.biases)]))
        # quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(rho(pre),W),rho(post))) for pre,W,post in zip(batch_units[:-1],self.weights,batch_units[1:])]))
        squared_norm = torch.sum(torch.stack([torch.sum(layer**2) for layer in batch_units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(rho(layer),b.view(-1, 1))) for layer,b in zip(batch_units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(torch.unsqueeze(pre, 2), torch.unsqueeze(post, 1)).view(batch_size, -1), W.view(-1, 1)))
                                     for pre,W,post in zip(batch_units[:-1],self.weights,batch_units[1:])]))
        return squared_norm + linear_terms + quadratic_terms

    # COST FUNCTION, DENOTED BY C
    def __cost(self, ground_truth, batch_size):
        batch_output = self.output[0:batch_size]
        return torch.sum((batch_output - ground_truth) ** 2)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, beta, ground_truth, batch_size):
        if beta == 0:
            return self.__energy(batch_size)
        else:
            return self.__energy(batch_size) + beta * self.__cost(ground_truth, batch_size)

    # def __unit_coorelations(self):
    #     weight_coorelations = [torch.mm(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
    #     bias_coorelations = [rho(t) for t in self.units]
    #     return weight_coorelations, bias_coorelations

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __predict_and_measure(self, ground_truth, batch_size):
        E = self.__energy(batch_size)
        C = self.__cost(ground_truth, batch_size)
        # y_softmax = F.log_softmax(self.output, dim=0)
        # return y_softmax, E, C
        return self.output[0:batch_size], E, C

    # Coverges network towards fixed point.
    def forward(self, input, num_iterations=None, beta=0, ground_truth=None, retain_graph=False):
        if self.model_version == 1:
            if not num_iterations:
                num_iterations = self.hyperparameters["num_iterations"]
            batch_size = input.shape[0]
            self.input[0:batch_size] = input
            for _ in range(num_iterations):
                self.energy_optimizer.zero_grad()
                energy = self.__total_energy(beta, ground_truth, batch_size)
                energy.backward(retain_graph=retain_graph)
                self.energy_optimizer.step()
            return self.output[0:batch_size]
        else:
            raise NotImplementedError
            # if beta == 0:
            #     self.input = torch.from_numpy(input).float()
            #     self.hidden = torch.mm(self.input, self.weights[0]).view(-1) + self.biases[1]
            #     self.hidden = rho(self.hidden)
            #     self.output = torch.mm(self.hidden.view(1, -1), self.weights[1]).view(-1) + self.biases[2]
            #     self.units = [self.input, self.hidden, self.output]
            # else:
            #     self.input = torch.from_numpy(input).float()
            #     self.hidden = torch.mm(self.input.view(1, -1), self.weights[0]).view(-1) + self.biases[1]
            #     self.hidden = rho(self.hidden)
            #     self.output = torch.mm(self.hidden.view(1, -1), self.weights[1]).view(-1) + self.biases[2]
            #     self.units = [self.input, self.hidden, self.output]
            # return self.output

    # Does contrastive parameter optimization. Change to Hebbian later.
    def optimize(self, input, ground_truth):
        if self.model_version == 1:
            batch_size = input.shape[0]
            self.forward(input, self.hyperparameters["num_iterations"], beta=0, ground_truth=None, retain_graph=True)  # Free Phase
            free_energy = self.__total_energy(0, ground_truth, batch_size)
            y_pred, mean_free_energy, mean_free_cost = self.__predict_and_measure(ground_truth, batch_size)
            self.forward(input, self.hyperparameters["num_iterations_neg"], beta=self.hyperparameters["beta"], ground_truth=ground_truth, retain_graph=True)  # Constrained Phase
            constrained_energy = self.__total_energy(self.hyperparameters["beta"], ground_truth, batch_size)
            self.model_optimizer.zero_grad()
            param_loss = (constrained_energy - free_energy) / self.hyperparameters["beta"]
            param_loss.backward()
            self.model_optimizer.step()
            return y_pred, mean_free_energy, mean_free_cost
        else:
            raise NotImplementedError
            # self.forward(input, n_iterations_pos, beta=0, ground_truth=None)  # Free Phase
            # free_wc, free_bc = self.__unit_coorelations()
            # y_pred, mean_free_energy, mean_free_cost = self.__predict_and_measure(ground_truth)
            # self.forward(input, n_iterations_neg, beta=beta, ground_truth=ground_truth)  # Constrained Phase
            # constrained_wc, constrained_bc = self.__unit_coorelations()
            # with torch.no_grad():
            #     self.weights = [layer + self.hyperparameters["alpha"] * (constrained - free) 
            #                     for layer, constrained, free in zip(self.weights, constrained_wc, free_wc)]
            #     self.biases = [layer + self.hyperparameters["alpha"] * (constrained - free)  
            #                 for layer, constrained, free in zip(self.biases, constrained_bc, free_bc)]
            # return y_pred, mean_free_energy, mean_free_cost

    def copy(self):
        new_net = Equilibrium_Propagation_Network(self.hyperparameters["input_size"], self.hyperparameters["output_size"], 
                                                  self.hyperparameters["num_hidden"], self.hyperparameters["batch_size"], 
                                                  self.hyperparameters["energy_learn_rate"], self.hyperparameters["param_learn_rate"], 
                                                  self.hyperparameters["num_iterations"], self.hyperparameters["num_iterations_neg"],
                                                  self.hyperparameters["beta"])
        new_net.input = torch.tensor(self.input)
        new_net.hidden = torch.tensor(self.hidden)
        new_net.output = torch.tensor(self.output)
        new_net.units = [new_net.input, new_net.hidden, new_net.output]
        new_net.free_units = [new_net.hidden, new_net.output]
        new_net.biases = [torch.tensor(t) for t in self.biases]
        new_net.weights = [torch.tensor(t) for t in self.weights]
        new_net.params = new_net.biases + new_net.weights
        # new_net.energy_optimizer = optim.Adam(new_net.free_units, lr=self.hyperparameters["energy_learn_rate"])
        # new_net.model_optimizer = optim.Adam(new_net.params, lr=self.hyperparameters["param_learn_rate"])
        return new_net

    def copy_(self, source_net):
        self.input = torch.tensor(source_net.input)
        self.hidden = torch.tensor(source_net.hidden)
        self.output = torch.tensor(source_net.output)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        self.biases = [torch.tensor(t) for t in source_net.biases]
        self.weights = [torch.tensor(t) for t in source_net.weights]
        self.params = self.biases + self.weights