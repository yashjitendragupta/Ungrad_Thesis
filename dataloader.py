import scipy.io
import numpy as np
import os 
from torch.utils.data import Dataset
import torch


class FR_Dataset(Dataset):
    def __init__(self):
        # set directory for set
        int_set_directory = os.getcwd() + '/test_set/freq_integrated/'
        list_set_directory = os.getcwd() + '/test_set/freq_listener/'
        int_set = os.listdir(int_set_directory)
        list_set = os.listdir(list_set_directory)
        self.length = len(int_set)
        print("data_found")

        # Load data from sets
        input_set = np.zeros((self.length,256))
        output_set = np.zeros((self.length,256))
        for i in range(self.length):
            mat = scipy.io.loadmat(int_set_directory + int_set[i])
            input_set[i] = np.concatenate((mat['h'].flatten(), np.zeros((6))))
        for i in range(self.length):
            mat = scipy.io.loadmat(list_set_directory + list_set[i])
            output_set[i] = np.concatenate((mat['h'].flatten(), np.zeros((6))))

        # create tensors
        self.X = torch.from_numpy(input_set).to("cuda:0")
        self.Y = torch.from_numpy(output_set).to("cuda:0")

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        return self.X[idx]
