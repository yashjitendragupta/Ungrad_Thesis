import torch
import torch.nn as nn
import torch.cuda
import scipy.io
import numpy as np
import os

# Parameters

num_epochs = 10

# set directory for set
int_set_directory = os.getcwd() + '/test_set/freq_integrated/'
list_set_directory = os.getcwd() + '/test_set/freq_listener/'
int_set = os.listdir(int_set_directory)
list_set = os.listdir(list_set_directory)
print("data_found")

# Load data from sets
input_set = np.zeros((len(int_set),256))
output_set = np.zeros((len(list_set),256))
for i in range(len(int_set)):
    mat = scipy.io.loadmat(int_set_directory + int_set[i])
    input_set[i] = np.concatenate((mat['h'].flatten(), np.zeros((6))))
for i in range(len(list_set)):
    mat = scipy.io.loadmat(list_set_directory + list_set[i])
    output_set[i] = np.concatenate((mat['h'].flatten(), np.zeros((6))))

# create tensors
X = torch.from_numpy(input_set).to("cuda:0")
Y = torch.from_numpy(output_set).to("cuda:0")


# Create model
# look into CNN's
maxpool_2     = torch.nn.MaxPool1d(2)
upsample_2    = torch.nn.Upsample(scale_factor=2, mode='nearest')
enc_conv_1    = torch.nn.Conv1d(len(int_set),len(int_set),15, device = "cuda:0", dtype=X.dtype, padding = "same")
enc_conv_2    = torch.nn.Conv1d(len(int_set),len(int_set),15, device = "cuda:0", dtype=X.dtype, padding = "same")
enc_conv_3    = torch.nn.Conv1d(len(int_set),len(int_set),15, device = "cuda:0", dtype=X.dtype, padding = "same")
bottom_conv   = torch.nn.Conv1d(len(int_set),len(int_set),15, device = "cuda:0", dtype=X.dtype, padding = "same")
dec_conv_1    = torch.nn.Conv1d(len(int_set),len(int_set),5 , device = "cuda:0", dtype=X.dtype, padding = "same")
dec_conv_2    = torch.nn.Conv1d(len(int_set),len(int_set),5 , device = "cuda:0", dtype=X.dtype, padding = "same")
dec_conv_3    = torch.nn.Conv1d(len(int_set),len(int_set),5 , device = "cuda:0", dtype=X.dtype, padding = "same")






# specify loss fn
# look at UNET architecture
loss_fn = nn.MSELoss()

# Specify optimizer
optimizer = torch.optim.Adam(list(enc_conv_1.parameters())+list(enc_conv_2.parameters())+list(enc_conv_3.parameters())+\
                             list(dec_conv_1.parameters())+list(dec_conv_2.parameters())+list(dec_conv_3.parameters()))
                             


# run model
for n in range(num_epochs):
    # encoder layer 1
    enc_1 = enc_conv_1(X)
    enc_1_down = maxpool_2(enc_1)

    # encoder layer 2
    enc_2 = enc_conv_2(enc_1_down)
    enc_2_down = maxpool_2(enc_2)

    # encoder layer 2
    enc_3 = enc_conv_3(enc_2_down)
    enc_3_down = maxpool_2(enc_3)

    # bottom layer
    dec_3_down = bottom_conv(enc_3_down)
    dec_3 = upsample_2

    print(dec_3_down.size())









    loss = loss_fn(enc_1, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
