from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import pickle
import numpy as np
from torchvision.transforms import ToTensor
from datetime import datetime

# Inherited class of neural network (NN)
# You should not change it unless you want to experiment with other NN parameters
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train_htd(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #what if we don't use batches for training?
    for data_img, target in train_loader:
        # print("data =", data)
        data = ToTensor()(data_img).unsqueeze(0)
        print("type(data) =", type(data))

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("type(data) =", type(data))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("data =", data)
        # print("type(data) =", type(data))
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     #     100. * batch_idx / len(train_loader), loss.item()))
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch,
        #         batch_idx * len(data),
        #         len(train_loader),
        #         100. * batch_idx / len(train_loader),
        #         loss.item()
        #     ))
        #
        #     if args.dry_run:
        #         break

def test_htd(model, device, test_loader, n_tot):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_img, target in test_loader:
            data = ToTensor()(data_img).unsqueeze(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n_tot

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n_tot,
        100. * correct / n_tot))

def test(model, device, test_loader, n_tot):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        # print("\n Expected data structure:")
        # print("type(test_loader) =", type(test_loader))
        # print("type(test_loader[0]) =", type(test_loader[0]))
        # print("type(test_loader[0][0]) =", type(test_loader[0][0]))
        # print("type(test_loader[0][1]) =", type(test_loader[0][1]))
        # print("test_loader[0][0].shape =", test_loader[0][0].shape)
        # print("test_loader[0][1].shape =", test_loader[0][1].shape)

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n_tot

    print('\nTest result: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, n_tot,
        100. * correct / n_tot))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    #
    # The following is just the original implementaion of NN-based classification tasks (with different data sets)
    #

    ## DFZ: for MNIST (70% -> 78%)
    datasetname = "MNIST"
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    ## DFZ: for KMNIST (34% -> 47%)
    # datasetname = "KMNIST"
    # dataset1 = datasets.KMNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.KMNIST('../data', train=False,
    #                    transform=transform)

    ## DFZ: for FashionMNIST (63% -> 58%)
    # datasetname = "FashionMNIST"
    # dataset1 = datasets.FashionMNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.FashionMNIST('../data', train=False,
    #                    transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    # Limit the training data to 1,000; batch size is default to 100 now, see above
    # So, for training data, there's a prefix batch ID;
    #       for test data howefver, no such batch ID exists.
    train_loader_htd = []
    nsamples = 10
    # for batch_idx, (data, target) in enumerate(train_loader):
    for i, (data, target) in enumerate(train_loader):
        if i >= nsamples:
            break
        # print("i =", i)
        train_loader_htd.append((data, target))
    # print(train_loader_htd[0][1][0].shape)
    # print(len(train_loader_htd[0]))
    # exit(0)

    # Limit the test data to 200
    test_loader_htd = []
    nsamples = 2
    i = 0
    # for batch_idx, (data, target) in enumerate(train_loader):
    for data, target in test_loader:
        if i >= nsamples:
            break
        # print("i =", i)
        test_loader_htd.append((data, target))
        i = i + 1
    # print(test_loader_htd[0][1].shape)
    # print(len(test_loader_htd[0]))

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):

        print("Start PyTorch training for", datasetname)
        train(args, model, device, train_loader_htd, optimizer, epoch)

        print("Start PyTorch testing for", datasetname)
        test(model, device, test_loader_htd, len(test_loader_htd) * len(test_loader_htd[0][1]))
        scheduler.step()

    #
    # Now let's start the training and testing for the topologically-denosed data
    #

    # Directory for HTD data
    htd_data_dir = './htd_data/'
    # if not os.path.exists(htd_data_dir):
    #     os.mkdir(htd_data_dir)

    # Load data from pickle
    len_training = 1000
    htd_trainset = htd_data_dir + datasetname + '_htd_trainset.pkl'
    training_data_htd = pickle.load(open(htd_trainset, 'rb'))
    # print("type(loaded_data) =", type(loaded_data))
    print("Loaded", len(training_data_htd), "HTD training figures.")
    len_test = 200
    htd_testset = htd_data_dir + datasetname + '_htd_testset.pkl'
    test_data_htd = pickle.load(open(htd_testset, 'rb'))
    # print("type(loaded_data) =", type(loaded_data))
    print("Loaded", len(test_data_htd), "HTD test figures.")

    # HTD training
    print("Start HTD training for", datasetname)
    epoch = 1 # just run once

    training_data_htd_batch = list2batch(training_data_htd, 100)
    train(args, model, device, training_data_htd_batch, optimizer, epoch)

    print("Start HTD testing for", datasetname)
    test_data_htd_batch = list2batch(test_data_htd, 100)
    test(model, device, test_data_htd_batch, len(test_data_htd))

'''
Convert a list ((image, label)) into batches
'''
def list2batch(lst, bsize):
    # print("\n Input data structure:")
    # print(type(lst))
    # print(type(lst[0]))
    # print(type(lst[0][0]))
    # print(type(lst[0][1]))

    idx = 0
    res = [] # result
    img_ary = np.zeros([bsize,1,28,28])
    label_ary = np.zeros([bsize], dtype=int)
    for img, label in lst:
        img_ary[idx % bsize][0] = np.array(img)
        label_ary[idx % bsize] = label
        idx = idx + 1
        if not (idx % bsize):
            #done with one batch
            img_tensor = torch.from_numpy(img_ary).float()
            label_tensor = torch.from_numpy(label_ary)
            res.append((img_tensor, label_tensor))
            img_ary = np.zeros([bsize, 1, 28, 28])
            label_ary = np.zeros([bsize], dtype=int)

    # print(type(res))
    # print(type(res[0]))
    # print(type(res[0][0]))
    # print(type(res[0][1]))
    # print(res[0][0].shape)
    # print(res[0][1].shape)
    #
    return res
    # pass
    # l = len(lst)
    # for ndx in range(0, l, bsize):
    #     yield lst[ndx:min(ndx + bsize, l)]

if __name__ == '__main__':
    main()
