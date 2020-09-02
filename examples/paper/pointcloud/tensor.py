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
