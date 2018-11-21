from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from hebbian_learning.models.equilibrium_propagation_baseline import Equilibrium_Propegation_Network

def train(args, model, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        one_hot = torch.zeros((10), dtype=torch.float32)
        target = one_hot.scatter_(0, target, 1)
        pred, energy, cost = model.optimize(data, args.n_iterations_pos, args.n_iterations_neg, args.beta, target)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCost, Energy: {:.6f} | {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cost.item(), energy.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model.forward(data, args.n_iterations)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epsilon', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--beta', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n_iterations_pos', type=int, default=20, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n_iterations_neg', type=int, default=4, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

    model = Equilibrium_Propegation_Network(28*28, 10, args.epsilon, args.alpha)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, epoch)
        test(args, model, test_loader)

if __name__ == '__main__':
    main()