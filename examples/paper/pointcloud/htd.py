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
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# start working on mnist data
import torchvision
import torchvision.datasets as datasets
from PIL import Image

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# index = 3000
# x, y = mnist_trainset[index]
# x.show()
# print(type(x))

# print(type(y))
# print("y =", y)
# print(len(mnist_testset))

##################################
###### The following code shows that MNIST data are not stored in alphanumerical order
##################################
# for i in range(len(mnist_trainset)):
#     if 0 == i % 3000:
#         _, y_i =  mnist_trainset[i]
#         print("i = ", i, "y_i = ", y_i)
# print("So, trainset's labels are random.")
# for i in range(len(mnist_testset)):
#     if 0 == i % 500:
#         _, y_i =  mnist_testset[i]
#         print("i = ", i, "y_i = ", y_i)
# print("So, testset's labels are random.")

#########################################
# If you want to see the rectified figure
#########################################
# data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x))))
# print(type(data_greylevel))
# print(data_greylevel.shape)
# print(data_greylevel)

############################
#convert a grey-level image into list of binary dots
############################
# k_tot = np.count_nonzero(data_greylevel)
# data = np.zeros(shape=(k_tot,2))
# len = 28 #the MNIST images are always 28*28
# k = 0
# for i in range(len):
#     for j in range(len):
#         if data_greylevel[i][j] != 0:
#             data[k] = [i,j]
#             k = k + 1
# print(data.shape)
# print("data = ", data)
# data = np.asarray(np.nonzero(data_greylevel)).T

# folder for storing figures of high-order topological denoising
figfolder = 'htd_figures/'
if not os.path.exists(figfolder):
    os.mkdir(figfolder)

# seed for reproduciblity
# np.random.seed(0)

# fig=plt.figure(figsize=(5,5))
# plt.scatter(data[:,0], data[:,1])
# plt.axis('equal')
# plt.axis('off')
# plt.savefig(figfolder + 'uniform.png')
# plt.close(fig)

#
# # alpha layer
# layer = AlphaLayer(maxdim=1)

'''
Preprocess the image with topological denoising
'''
def topo_reduce(input, steps=100):
    x = torch.autograd.Variable(torch.tensor(input).type(torch.float), requires_grad=True)
    lr = 1e-2
    optimizer = torch.optim.Adam([x], lr=lr)
    layer = AlphaLayer(maxdim=1)
    f2 = BarcodePolyFeature(0, 2, 0)
    f3 = BarcodePolyFeature(1, 2, 1) # 1-dim
    f4 = BarcodePolyFeature(0, 2, 0) # 0-dim
    for i in range(steps):
        optimizer.zero_grad()

        dgminfo = layer(x) #but does layer(x) change over the iterations?
        # print("Increase H1, decrease H0")
        # loss = -f3(dgminfo) + f4(dgminfo)
        loss = -f3(dgminfo) * 1 + f4(dgminfo) * 20
        loss.backward()

        optimizer.step()

        if loss.item() <= 5.0:  # should have some eplison to avoid overfitting
            break
        if (i % 50 == 0):
            print("[Iter %d] [top loss: %f]" % (i, loss.item()))
    return x

# x = topo_reduce(data)
# print("x.type =", x.type)
# y = x.detach().numpy().astype(int)

# Directory for HTD data
htd_data_dir = './htd_data/'
if not os.path.exists(htd_data_dir):
    os.mkdir(htd_data_dir)

# HTD test set
len = len(mnist_testset)
len = 200
testset_htd = []
print("HTD test set started at", datetime.now().strftime("%H:%M:%S"))
for i in range(len):
    print("\n==========================================")
    print("Processing the " + str(i) + "-th figure.")
    x_i, y_i = mnist_testset[i]
    # print("type(x_i) =", type(x_i))

    # # Show the original figure
    # data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x_i))))
    # data = np.asarray(np.nonzero(data_greylevel)).T
    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.axis('equal')
    # plt.axis('off')
    # plt.savefig(figfolder + 'origin_' + str(i) + '.png')
    # plt.close(fig)

    x_i_2d = np.asarray(np.nonzero(x_i)).T
    x_i_2d_htd = topo_reduce(x_i_2d).detach().numpy().astype(int)
    x_i_htd = np.zeros((28, 28))
    x_i_htd[tuple(x_i_2d_htd.T)] = 1

    # # Show the denoised figure
    # fig = plt.figure(i)
    # plt.imshow(x_i_htd)
    # plt.savefig(figfolder + "htd_" + str(i) + ".png")
    # plt.close(fig)

    print("Reduced pixels from", int(np.count_nonzero(np.array(x_i))), "to", np.count_nonzero(x_i_htd))

    # Convert trainset_htd to Image and save as the new training set
    new_im = Image.fromarray(x_i_htd)
    # new_im.convert("L").save(figfolder + "tmp_" + str(i) + ".png")
    testset_htd.append((new_im, y_i))
print("\nHTD stoped at", datetime.now().strftime("%H:%M:%S"))
htd_testset = htd_data_dir + 'htd_testset.pkl'
file = open(htd_testset, 'wb')
pickle.dump(testset_htd, file)

# HTD training set
len = len(mnist_trainset)
len = 1000
trainset_htd = []
print("HTD training set started at", datetime.now().strftime("%H:%M:%S"))
for i in range(len):
    print("\n==========================================")
    print("Processing the " + str(i) + "-th figure.")
    x_i, y_i = mnist_trainset[i]
    # print("type(x_i) =", type(x_i))

    # Show the original figure
    data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x_i))))
    data = np.asarray(np.nonzero(data_greylevel)).T
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1])
    plt.axis('equal')
    plt.axis('off')
    # plt.savefig(figfolder + 'origin_' + str(i) + '.png')
    plt.close(fig)

    x_i_2d = np.asarray(np.nonzero(x_i)).T
    x_i_2d_htd = topo_reduce(x_i_2d).detach().numpy().astype(int)
    x_i_htd = np.zeros((28, 28))
    x_i_htd[tuple(x_i_2d_htd.T)] = 1

    # Show the denoised figure
    fig = plt.figure(i)
    plt.imshow(x_i_htd)
    # plt.savefig(figfolder + "htd_" + str(i) + ".png")
    plt.close(fig)

    print("Reduced pixels from", int(np.count_nonzero(np.array(x_i))), "to", np.count_nonzero(x_i_htd))

    # Convert trainset_htd to Image and save as the new training set
    new_im = Image.fromarray(x_i_htd)
    # new_im.convert("L").save(figfolder + "tmp_" + str(i) + ".png")
    trainset_htd.append((new_im, y_i))
print("\nHTD stoped at", datetime.now().strftime("%H:%M:%S"))

# Save htd data into pickle files
htd_trainset = htd_data_dir + 'htd_trainset.pkl'
file = open(htd_trainset, 'wb')
pickle.dump(trainset_htd, file)

# # Load data from pickle
# loaded_data = pickle.load(open(htd_trainset, 'rb'))
# print("type(loaded_data) =", type(loaded_data))
# # Compare unpickled and original data
# unpickled = np.array(loaded_data[len-1][0])
# original = np.array(trainset_htd[len-1][0])
# for i in range(28):
#     for j in range(28):
#         if unpickled[i][j] * original[i][j] > 0.5:
#             print("( i =", i, "j =", j, ":", unpickled[i][j], " ?= ", original[i][j])


# # Compare the denoised data and the originals (the last one)
# image_original = np.array(mnist_trainset[len-1][0])
# image_htd = np.array(trainset_htd[len-1][0])
# # print(type(data))
# for i in range(28):
#     for j in range(28):
#         print("( i =", i, "j =", j, ":", image_original[i][j], " ?= ", image_htd[i][j])



# plt.figure(figsize=(5,5))
# plt.scatter(y[:,0], y[:,1])
# plt.axis('equal')
# plt.axis('off')
# plt.savefig(figfolder + 'uniform+H1-H0.png')

#convert the indices into a matrix of one's
# ma = np.zeros((28, 28))
# ma[tuple(y.T)] = 1
# # print("ma = ", ma)
# plt.figure(figsize=(5,5))
# ma_rot90 = np.rot90(np.array(ma))
# plt.imshow(ma_rot90);
# plt.savefig(figfolder + 'uniform_matrix.png')
#
# print("reduce pixels from", int(data.size/2), "to", np.count_nonzero(ma_rot90))