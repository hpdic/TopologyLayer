import htd
import nn_htd
import os
import time

#Parameters:
mydataset = 1 # TODO: I think we only have time for this dataset (MNIST)
if 1 == mydataset:
    mydataset_name = 'MNIST'
elif 2 == mydataset:
    mydataset_name = 'KMNIST'
elif 3 == mydataset:
    mydataset_name = 'FashionMNIST'
else:
    print("Unknown dataset.")
    exit(0)
myhtdflag = 1 # Don't change this unless you want to check the original NN; but that can be done at nn_htd.py
fulllist = [1, 2, 4, 8, 16]
weights_dim1 = [-2]
weights_dim0 = [2]

#Results on file:
res_folder = 'eval_results/'
if not os.path.exists(res_folder):
    os.mkdir(res_folder)
fd = open(res_folder + "summary_" + mydataset_name + ".csv", "w")
fd.write("id, " + "seconds, " + "dim1, " + "dim0, " + "n_pixel_before, " + "n_pixel_after, " +
         "epoch1, " + "epoch2, " + "epoch3" + "\n")

#Sweep the space
n_id = 0
for dim1 in weights_dim1:
    for dim0 in weights_dim0:

        n_id += 1
        fd.write(str(n_id))
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print("Experiment ID = " + str(n_id) + "\n")

        # HTD:
        tic = time.perf_counter()
        (total_pixel_before, total_pixel_after) = htd.topo(mydataset, dim1, dim0)
        toc = time.perf_counter()

        # #Summary
        # print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        # print("dim1 weight = " + str(weight_dim1))
        # print("dim0 weight = " + str(weight_dim0))
        # print(os.path.basename(__file__) + ": total_pixel_after = " + str(total_pixel_after))
        # print(os.path.basename(__file__) + ": total_pixel_before = " + str(total_pixel_before))
        # print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        fd.write(f", {toc-tic: .3f}, " + str(dim1) +  ", " + str(dim0) +
                 ", " + str(total_pixel_before) + ", " + str(total_pixel_after))

        # NN
        accuracy = nn_htd.nn_topo(mydataset, myhtdflag, dim1, dim0)

        for i in range(len(accuracy)):
            # print("Epoch #" + str(i) + ": " + str(accuracy[i]))
            fd.write(", " + str(accuracy[i]))
        fd.write("\n")
        fd.flush() #Don't wait until the very end for persistence

fd.close()
