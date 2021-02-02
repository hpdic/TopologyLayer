import htd
import nn_htd
import os

mydataset = '1'
myhtdflag = 1
weights_dim1 = [1,2]
weights_dim0 = [4,8]

for weight_dim1 in weights_dim1:
    for weight_dim0 in weights_dim0:

        # #HTD:
        # (total_pixel_before, total_pixel_after) = htd.topo(mydataset, weight_dim1, weight_dim0)
        #
        # #Summary
        # print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        # print("dim1 weight = " + str(weight_dim1))
        # print("dim0 weight = " + str(weight_dim0))
        # print(os.path.basename(__file__) + ": total_pixel_after = " + str(total_pixel_after))
        # print(os.path.basename(__file__) + ": total_pixel_before = " + str(total_pixel_before))
        # print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

        #TODO: NN
        accuracy = nn_htd.nn_topo(dataset=mydataset, htdflag=myhtdflag)
        for i in range(len(accuracy)):
            print("Epoch #" + str(i) + ": " + str(accuracy[i]))
