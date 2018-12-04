# import cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def rho(s):
    return torch.clamp(s,0.,1.)
    # return torch.sigmoid(s)

# MLP Model. Might want to do a vanilla RNN later.
class Equilibrium_Propagation_Value_Network(nn.Module):
    def __init__(self, input_size, output_size, num_hidden, energy_learn_rate, param_learn_rate, num_iterations, num_iterations_neg, beta):
        super(Equilibrium_Propagation_Value_Network, self).__init__()

        # self.path = name+".save"
        self.hyperparameters = {}
        self.hyperparameters["input_size"] = input_size
        self.hyperparameters["output_size"] = output_size
        self.hyperparameters["num_hidden"] = num_hidden
        self.hyperparameters["energy_learn_rate"] = energy_learn_rate
        self.hyperparameters["param_learn_rate"] = param_learn_rate
        self.hyperparameters["num_iterations"] = num_iterations
        self.hyperparameters["num_iterations_neg"] = num_iterations_neg
        self.hyperparameters["beta"] = beta
        
        self.input = torch.zeros([input_size], dtype=torch.float32, requires_grad=False)
        self.hidden = torch.zeros([num_hidden], dtype=torch.float32, requires_grad=True)
        self.output = torch.zeros([output_size], dtype=torch.float32, requires_grad=True)
        self.units = [self.input, self.hidden, self.output]
        self.free_units = [self.hidden, self.output]
        
        self.biases = [torch.zeros(t.shape, requires_grad=False) for t in self.units]
        self.weights = [nn.init.xavier_uniform_(torch.zeros((pre_t.shape[0], post_t.shape[0]), requires_grad=False)) for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
        self.params = self.biases + self.weights

        self.energy_optimizer = optim.Adam(self.free_units, lr=energy_learn_rate)
        self.model_optimizer = optim.Adam(self.params, lr=param_learn_rate)

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, batch_size):
        squared_norm = torch.sum(torch.stack([torch.sum(layer**2) for layer in self.units])) / 2.
        linear_terms = -torch.sum(torch.stack([torch.sum(torch.dot(rho(layer),b)) for layer,b in zip(batch_units,self.biases)]))
        quadratic_terms = -torch.sum(torch.stack([torch.sum(torch.matmul(torch.matmul(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1)).view(batch_size, -1), W.view(-1, 1)))
                                     for pre,W,post in zip(self.units[:-1],self.weights,self.units[1:])]))
        return squared_norm + linear_terms + quadratic_terms

    # COST FUNCTION, DENOTED BY C
    def __cost(self, ground_truth, batch_size):
        return torch.sum((self.output - ground_truth) ** 2)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, beta, ground_truth, batch_size):
        if beta == 0:
            return self.__energy(batch_size)
        else:
            return self.__energy(batch_size) + beta * self.__cost(ground_truth, batch_size)

    def __unit_coorelations(self):
        weight_coorelations = [torch.matmul(rho(pre_t).view(-1, 1), rho(post_t).view(1, -1))for pre_t, post_t in zip(self.units[:-1],self.units[1:])]
        bias_coorelations = [rho(t) for t in self.units]
        return weight_coorelations, bias_coorelations

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __predict_and_measure(self, ground_truth, batch_size):
        E = self.__energy(batch_size)
        C = self.__cost(ground_truth, batch_size)
        # y_softmax = F.log_softmax(self.output, dim=0)
        # return y_softmax, E, C
        return self.output[0:batch_size], E, C

    def __reset_state(self):
        with torch.no_grad():
            self.input.zero_()
            self.hidden.zero_()
            self.output.zero_()
            # self.units = [self.input, self.hidden, self.output]
            # self.free_units = [self.hidden, self.output]

    # Coverges network towards fixed point.
    def forward(self, input=None, batch_size=None, num_iterations=None, beta=0, ground_truth=None, retain_graph=False):
        if not num_iterations:
            num_iterations = self.hyperparameters["num_iterations"]
        if input is not None:
            self.__reset_state()
            self.batch_size = input.shape[0]
            self.input[0:self.batch_size] = input
        for i in range(num_iterations):
            self.energy_optimizer.zero_grad()
            energy = self.__total_energy(beta, ground_truth, self.batch_size)
            energy.backward(retain_graph=retain_graph)
            self.energy_optimizer.step()
        return self.output[0:self.batch_size]

    # Requires running free phase before calling this. eval is unused, just to satisfy interface requirment.
    def forward_and_optimize(self, input, ground_truth):
        if self.model_version == 1:
            if np.random.uniform() < 0.5:
                self.hyperparameters["beta"] *= -1
            batch_size = input.shape[0]
            self.forward(input=input, num_iterations=self.hyperparameters["num_iterations"], beta=0, ground_truth=None)  # Free Phase
            free_energy = self.__total_energy(0, ground_truth, self.batch_size)
            y_pred, mean_free_energy, mean_free_cost = self.__predict_and_measure(ground_truth, self.batch_size)
            self.forward(batch_size=self.batch_size, num_iterations=self.hyperparameters["num_iterations_neg"], 
                         beta=self.hyperparameters["beta"], ground_truth=ground_truth)  # Constrained Phase
            constrained_energy = self.__total_energy(self.hyperparameters["beta"], ground_truth, self.batch_size)
            self.model_optimizer.zero_grad()
            param_loss = (constrained_energy - free_energy) / self.hyperparameters["beta"]
            param_loss.backward() 
            self.model_optimizer.step()
            return y_pred, mean_free_energy, mean_free_cost

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