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
import sys

def main():
    argc = len(sys.argv)
    argv1 = '1' if argc < 2 else argv[1]
    argv2 = '1' if argc < 3 else argv[2]
    argv3 = '1' if argc < 4 else argv[3]

    topo(argv1, int(argv2), int(argv3))
    # topo(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

def topo(*args):

    if args is not None:
        whichdataset = int(args[0])
        g_dim1wt = int(args[1])
        g_dim0wt = int(args[2])

    show_fig = False

    # We have tested up to 1000 and 200 for initial experiments, which took too much time; we now use (300, 100)
    len_training = 300    # 1 for trivial test; 60000 for fullsize; 300 for most experiments
    len_test = 100        # 1 for trivial test; 10000 for fullsize; 100 for most experiments
    htd_batch = 50

    total_pixel_after = 0
    total_pixel_before = 0

    #
    #DFZ: parse the dataset name
    #
    if "MNIST" == whichdataset or 1 == whichdataset:
        datasetname = "MNIST"
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    elif "KMNIST" == whichdataset or 2 == whichdataset:
        datasetname = "KMNIST"
        mnist_trainset = datasets.KMNIST('./data', train=True, download=True,
                                   transform=None)
        mnist_testset = datasets.KMNIST('./data', train=False,
                                   transform=None)

    elif "FashionMNIST" == whichdataset or 3 == whichdataset:
        datasetname = "FashionMNIST"
        mnist_trainset = datasets.FashionMNIST('./data', train=True, download=True,
                                               transform=None)
        mnist_testset = datasets.FashionMNIST('./data', train=False,
                                              transform=None)
    else:
        print("Data set not available; please choose from MNIST[1], KMINST[2], or FashionMNIST[3].")
        exit(0)

    # DFZ: MNIST
    # datasetname = "MNIST"
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    # mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    # # DFZ: KMNIST -- Japanese Character data, exactly the same format as MNIST
    # datasetname = "KMNIST"
    # mnist_trainset = datasets.KMNIST('./data', train=True, download=True,
    #                            transform=None)
    # mnist_testset = datasets.KMNIST('./data', train=False,
    #                            transform=None)

    # # DFZ: FashionMNIST -- Fashion Clothes in the format of MNIST
    # datasetname = "FashionMNIST"
    # mnist_trainset = datasets.FashionMNIST('./data', train=True, download=True,
    #                            transform=None)
    # mnist_testset = datasets.FashionMNIST('./data', train=False,
    #                            transform=None)


    ############################
    ##### The following shows a specific (x,y) data pair from the data set
    ############################
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
    #for now, we want to set the same weights on different dimensions
    if g_dim1wt == None:
        g_dim1wt = 1
    if g_dim0wt == None:
        g_dim0wt = 1

    # print("dim1 weight = " + str(g_dim1wt))
    # print("dim0 weight = " + str(g_dim0wt))

    # folder for storing experimental results of pixel compression of each figure
    pixel_folder = 'eval_results/'
    if not os.path.exists(figfolder):
        os.mkdir(figfolder)

    #
    #DFZ: save the compression rate result in a seperate file
    #
    fd = open(pixel_folder+"res_compress_rate_"+datasetname+"_d1_"+str(g_dim1wt)+"_d0_"+str(g_dim0wt)+".csv", "w")

    def topo_reduce(input, steps=100):
        x = torch.autograd.Variable(torch.tensor(input).type(torch.float), requires_grad=True)
        lr = 1e-2
        optimizer = torch.optim.Adam([x], lr=lr)
        layer = AlphaLayer(maxdim=1)
        f2 = BarcodePolyFeature(0, 2, 0) # any usefulness?
        f3 = BarcodePolyFeature(1, 2, 1) # 1-dim
        f4 = BarcodePolyFeature(0, 2, 1) # 0-dim
        for i in range(steps):
            optimizer.zero_grad()

            dgminfo = layer(x) #but does layer(x) change over the iterations? Yes, because x is updated (due to optimizer)

            # print(type(x),len(x), len(x[0]),x[0][0])
            # print(type(f3(dgminfo)))
            # print(f3(dgminfo))

            # print("Increase H1, decrease H0")
            # loss = -f3(dgminfo) + f4(dgminfo)
            loss = -f3(dgminfo) * g_dim1wt - f4(dgminfo) * g_dim0wt #not sure whether those weights will actually take effect

            # print("f3(dgminfo) = " + str(f3(dgminfo)))
            # print("f4(dgminfo) = " + str(f4(dgminfo)))

            loss.backward()

            optimizer.step()

            if loss.item() <= 5.0:  # should have some eplison to avoid overfitting
                break
            # if (i % 50 == 0):
            #     print("[Iter %d] [top loss: %f]" % (i, loss.item()))
        return x

    # x = topo_reduce(data)
    # print("x.type =", x.type)
    # y = x.detach().numpy().astype(int)

    # Directory for HTD data
    htd_data_dir = './htd_data/'
    if not os.path.exists(htd_data_dir):
        os.mkdir(htd_data_dir)

    # HTD test set
    # l = len(mnist_testset)
    testset_htd = []
    print("HTD started at", datetime.now().strftime("%H:%M:%S"))
    for i in range(len_test):
        # print("\n==========================================")
        if i % htd_batch == 0:
            print("Processing the " + str(i) + "-th test figure at " + datetime.now().strftime("%H:%M:%S"))
        x_i, y_i = mnist_testset[i]
        # print("type(x_i) =", type(x_i))

        # Show the original figure
        if show_fig:
            data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x_i))))
            data = np.asarray(np.nonzero(data_greylevel)).T
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(data[:, 0], data[:, 1])
            plt.axis('equal')
            plt.axis('off')
            plt.savefig(figfolder + 'origin_test_' + str(i) + '.png')
            plt.close(fig)

        x_i_2d = np.asarray(np.nonzero(x_i)).T
        x_i_2d_htd_raw = topo_reduce(x_i_2d).detach().numpy().astype(int)

        # _x = x_i_2d_htd_raw[:,0]
        # print("_x =", _x)
        # _xx = np.piecewise(_x, [_x < 0, _x > 27], [0, 27, lambda _x: _x])
        # print("_xx =", _xx)
        # exit(0)

        # this is very important, as HTD might convert pixels out of the picture boundary
        # there are two ways: (i) remove the point (ii) place the pixel on the boundary.
        # we are taking the second appraoch for now
        x_i_2d_htd = np.piecewise(x_i_2d_htd_raw, [x_i_2d_htd_raw < 0, x_i_2d_htd_raw > 27],
                                  [0, 27, lambda x_i_2d_htd_raw: x_i_2d_htd_raw])
        x_i_htd = np.zeros((28, 28))
        x_i_htd[tuple(x_i_2d_htd.T)] = 1

        # Show the denoised figure
        # fig = plt.figure(i)
        # plt.imshow(x_i_htd)
        if show_fig:
            x_i_2d_htd_up = np.asarray(np.nonzero(np.rot90(np.rot90(np.rot90(x_i_htd))))).T
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(x_i_2d_htd_up[:, 0], x_i_2d_htd_up[:, 1])
            plt.axis('equal')
            plt.axis('off')
            plt.savefig(figfolder + "htd_test_" + str(i) + ".png")
            plt.close(fig)

        # print("Reduced pixels from", int(np.count_nonzero(np.array(x_i))), "to", np.count_nonzero(x_i_htd))
        fd.write("test " + str(i) + ", " + str(int(np.count_nonzero(np.array(x_i)))) + ", " +
                 str(np.count_nonzero(x_i_htd)) + "\n")

        total_pixel_after += np.count_nonzero(x_i_htd)
        total_pixel_before += int(np.count_nonzero(np.array(x_i)))
        # print("total_pixel_after = " + str(total_pixel_after))
        # print("total_pixel_before = " + str(total_pixel_before))

        # Convert trainset_htd to Image and save as the new training set
        new_im = Image.fromarray(x_i_htd)
        # new_im.convert("L").save(figfolder + "tmp_" + str(i) + ".png")
        testset_htd.append((new_im, y_i))
    # print("\nHTD stoped at", datetime.now().strftime("%H:%M:%S"))
    htd_testset = htd_data_dir + datasetname + '_htd_testset.pkl'
    file = open(htd_testset, 'wb')
    pickle.dump(testset_htd, file)
    file.close()

    # HTD training set
    # len = len(mnist_trainset)
    trainset_htd = []
    # print("HTD training set started at", datetime.now().strftime("%H:%M:%S"))
    for i in range(len_training):
        # print("\n==========================================")
        if i % htd_batch == 0:
            print("Processing the " + str(i) + "-th training figure at " + datetime.now().strftime("%H:%M:%S"))
        x_i, y_i = mnist_trainset[i]
        # print("type(x_i) =", type(x_i))

        # # Show the original figure
        if show_fig:
            data_greylevel = np.rot90(np.rot90(np.rot90(np.array(x_i))))
            data = np.asarray(np.nonzero(data_greylevel)).T
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(data[:, 0], data[:, 1])
            plt.axis('equal')
            plt.axis('off')
            plt.savefig(figfolder + 'origin_training_' + str(i) + '.png')
            plt.close(fig)

        x_i_2d = np.asarray(np.nonzero(x_i)).T
        x_i_2d_htd_raw = topo_reduce(x_i_2d).detach().numpy().astype(int)
        x_i_2d_htd = np.piecewise(x_i_2d_htd_raw, [x_i_2d_htd_raw < 0, x_i_2d_htd_raw > 27],
                                  [0, 27, lambda x_i_2d_htd_raw: x_i_2d_htd_raw])
        x_i_htd = np.zeros((28, 28))
        x_i_htd[tuple(x_i_2d_htd.T)] = 1

        # # Show the denoised figure
        if show_fig:
            fig = plt.figure(i)
            plt.imshow(x_i_htd)
            plt.savefig(figfolder + "htd_training_" + str(i) + ".png")
            plt.close(fig)

        # print("Reduced pixels from", int(np.count_nonzero(np.array(x_i))), "to", np.count_nonzero(x_i_htd))
        fd.write("train " + str(i) + ", " + str(int(np.count_nonzero(np.array(x_i)))) + ", " +
                 str(np.count_nonzero(x_i_htd)) + "\n")

        total_pixel_after += np.count_nonzero(x_i_htd)
        total_pixel_before += int(np.count_nonzero(np.array(x_i)))
        # print("total_pixel_after = " + str(total_pixel_after))
        # print("total_pixel_before = " + str(total_pixel_before))

        # Convert trainset_htd to Image and save as the new training set
        new_im = Image.fromarray(x_i_htd)
        # new_im.convert("L").save(figfolder + "tmp_" + str(i) + ".png")
        trainset_htd.append((new_im, y_i))
    print("HTD stoped at", datetime.now().strftime("%H:%M:%S"), "\n")

    # Save htd data into pickle files
    htd_trainset = htd_data_dir + datasetname + '_htd_trainset.pkl'
    file = open(htd_trainset, 'wb')
    pickle.dump(trainset_htd, file)
    file.close()

    return (total_pixel_before, total_pixel_after)
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

if __name__ == '__main__':
    main()