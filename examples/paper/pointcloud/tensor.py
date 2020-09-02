'''
credits go to: https://johnwlambert.github.io/pytorch-tutorial/
'''

import numpy as np
import torch

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
