#
# For denoising, we want to decrease H0 and promote the largest H1 
#   htd: High-order Topological Denoising
#

from __future__ import print_function
from topologylayer.nn import AlphaLayer
from topologylayer.nn import *

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# start working on mnist data
import torchvision
import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

index = 3000
x, y = mnist_trainset[index]
x.show()
# print(type(x))

# print(type(y))
# print("y =", y)
# print(len(mnist_testset))

for i in range(len(mnist_trainset)):
    if 0 == i % 3000:
        _, y_i =  mnist_trainset[i]
        print("i = ", i, "y_i = ", y_i)
print("So, trainset's labels are random.")


for i in range(len(mnist_testset)):
    if 0 == i % 500:
        _, y_i =  mnist_testset[i]
        print("i = ", i, "y_i = ", y_i)
print("So, testset's labels are random.")

data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x))))
# print(type(data_greylevel))
# print(data_greylevel.shape)
# print(data_greylevel)

#convert a grey-level image into list of binary dots
k_tot = np.count_nonzero(data_greylevel)
data = np.zeros(shape=(k_tot,2))
len = 28 #the MNIST images are always 28*28
k = 0
for i in range(len):
    for j in range(len):
        if data_greylevel[i][j] != 0:
            data[k] = [i,j]
            k = k + 1
print(data.shape)
# print(data)

# folder for storing figures of high-order topological denoising
figfolder = 'htd_figures/'
import os
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# seed for reproduciblity
# np.random.seed(0)

# initial pointcloud
# num_samples = 100
# data = np.random.rand(num_samples, 2)
# print(data.shape)

plt.figure(figsize=(5,5))
plt.scatter(data[:,0], data[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform.png')
#
# # alpha layer
# layer = AlphaLayer(maxdim=1)

"""
# Increase H1
print("Increase H1")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
f1 = BarcodePolyFeature(1,2,0)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = -f1(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform+H1.png')

# decrease H1
print("Decrease H1")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = f1(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform-H1.png')
"""

# decrease H0
# print("Decrease H0")
# x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
# print(x)
# print(type(x))
# layer = AlphaLayer(maxdim=0)
# f2 = BarcodePolyFeature(0,2,0)
# lr=1e-2
# optimizer = torch.optim.Adam([x], lr=lr)
# adjust = 2 #default adjuster
# for i in range(100):
#     optimizer.zero_grad()
#     loss = adjust * f2(layer(x))
#     loss.backward()
#     optimizer.step()
#     if loss.item() <= 2.0: # should have some eplison to avoid overfitting
#         break
#     if (0 == i % 5):
#         print ("[Iter %d] [top loss: %f]" % (i, loss.item()))
#
# y = np.rint(x.detach().numpy())
# print(y)
# print(y.shape)
# plt.figure(figsize=(5,5))
# plt.scatter(y[:,0], y[:,1])
# plt.axis('equal')
# plt.axis('off')
# plt.savefig(figfolder + 'adjust' + str(adjust) + '_uniform-H0.png')


"""
# increase H0
print("Increase H0")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
for i in range(100):
    optimizer.zero_grad()
    loss = -f2(layer(x))
    loss.backward()
    optimizer.step()
    if (i % 5 == 4):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy()
plt.figure(figsize=(5,5))
plt.scatter(y[:,0], y[:,1])
plt.axis('equal')
plt.axis('off')
plt.savefig(figfolder + 'uniform+H0.png')

"""
# Fun combination of H0, H1
print("Increase H1, decrease H0")
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
lr=1e-2
optimizer = torch.optim.Adam([x], lr=lr)
layer = AlphaLayer(maxdim=1)
f3 = BarcodePolyFeature(1,2,1)
f4 = BarcodePolyFeature(0,2,0)
for i in range(1000):
    optimizer.zero_grad()
    dgminfo = layer(x)
    # loss = -f3(dgminfo) + f4(dgminfo)
    loss = -f3(dgminfo)*.5 + f4(dgminfo)*2
    loss.backward()
    optimizer.step()
    # print(type(loss.item()))
    if loss.item() <= 5.0: # should have some eplison to avoid overfitting
        break
    if (i % 50 == 0):
        print ("[Iter %d] [top loss: %f]" % (i, loss.item()))

y = x.detach().numpy().astype(int)

# plt.figure(figsize=(5,5))
# plt.scatter(y[:,0], y[:,1])
# plt.axis('equal')
# plt.axis('off')
# plt.savefig(figfolder + 'uniform+H1-H0.png')

#convert the indices into a matrix of one's
ma = np.zeros((28, 28))
ma[tuple(y.T)] = 1
# print("ma = ", ma)
plt.figure(2)
ma_rot90 = np.rot90(np.array(ma))
plt.imshow(ma_rot90);
plt.savefig(figfolder + 'uniform_matrix.png')