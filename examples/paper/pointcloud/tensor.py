'''
credits go to: https://johnwlambert.github.io/pytorch-tutorial/
'''

import numpy as np
import torch
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

x = torch.Tensor([1., 2., 3.])
print(x) # Prints "tensor([1., 2., 3.])"
print(x.shape) # Prints "torch.Size([3])"
print(torch.ones(2,1)) # Prints "tensor([[1.],[1.]])"
print(torch.zeros_like(x)) # Prints "tensor([0., 0., 0.])"

# Alternatively, create a tensor by bringing it in from Numpy
y = np.array([0,1,2,3])
print(y) # Prints "[0 1 2 3]"
x = torch.from_numpy(y)
print(x) # Prints "tensor([0, 1, 2, 3])"

print(x.dtype) # Prints "torch.int64", currently 64-bit integer type
x = x.type(torch.FloatTensor)
print(x.dtype) # Prints "torch.float32", now 32-bit float
print(x.float()) # Still "torch.float32"
print(x.type(torch.DoubleTensor)) # Prints "tensor([0., 1., 2., 3.], dtype=torch.float64)"
print(x.type(torch.LongTensor)) # Cast back to int-64, prints "tensor([0, 1, 2, 3])"

x = torch.tensor([[1,2,3]])
y = torch.tensor([[2,2,2]])
print(x.shape) # Prints "torch.Size([1, 3])"
print(x.t().shape) # Take matrix transpose, prints torch.Size([3, 1])
print(y.shape) # Prints "torch.Size([1, 3])"
print(torch.mul(x,y)) # Elementwise multiplication of arrays, prints "tensor([[2, 4, 6]])"
print(x.t().mm(y).shape) # Outer product of (3,1) and (1,3), prints "torch.Size([3, 3])"
print(x.mm(y.t())) # Dot product of (1,3) and (3,1), prints "tensor([[12]])"
print(torch.mm(x,y.t())) # Same dot product/inner product, prints "tensor([[12]])""
print(torch.matmul(x,y.t())) # Identical to above, prints "tensor([[12]])"


#combining tensors
x = torch.tensor([[1,2,3]])
y = torch.tensor([[2,2,2]])
print(x.shape) # Prints torch.Size([1, 3]), x is a row vector
print(y.shape) # Prints torch.Size([1, 3]), y is a row vector
print(torch.stack([x,y]).shape) # prints torch.Size([2, 1, 3])
print(torch.cat([x,y]).shape) #  prints "torch.Size([2, 3])"


'''
credit goes to: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # outputting 32 convolutional features, with a square kernel size of 3
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # Second 2D convolutional layer, taking in the 32 input layers,
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      # First fully connected layer
      self.fc1 = nn.Linear(9216, 128)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

my_nn = Net()
print(my_nn)


# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()
result = my_nn(random_data)
print (result)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# dataset1 = datasets.MNIST('../data', train=True, download=True,
#                           transform=transform)
# dataset2 = datasets.MNIST('../data', train=False,
#                           transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
# test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

#
# Now we need to convert the images to dataloader for pytorch
#
from torch.utils.data import TensorDataset, DataLoader

my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

tensor_x = torch.Tensor(my_x) # transform to torch tensor
tensor_y = torch.Tensor(my_y)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset) # create your dataloader

print("looks good")
exit(0)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(1, args.epochs + 1):
    # train(args, model, device, train_loader, optimizer, epoch)
    # test(model, device, test_loader)
    # scheduler.step()
    pass

if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
