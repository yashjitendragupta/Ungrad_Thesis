

# imports:
import torch
import torch.nn as nn
import torch.cuda
import scipy.io
from scipy.io import wavfile
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm






# Parameter:
num_epochs = 2000
batch_size = 100
load_params = False
result_folder = ".\\Results\\FC_result\\"
test_folder = "\\Datasets\\test_set\\"
train_folder = "\\Datasets\\train_set\\"
validation_folder = "\\Datasets\\validation_set\\"
model_filename = "U_Net_dense.pth"

# Dataset Definition:
class Time_Dataset(Dataset):
    def __init__(self,folder_name):
        # set directory for set
        self.int_set_directory = os.getcwd() + folder_name + "\\time_integrated\\"
        self.list_set_directory = os.getcwd() + folder_name + '\\time_listener\\'
        self.int_set = os.listdir(self.int_set_directory)
        self.list_set = os.listdir(self.list_set_directory)
        self.length = len(self.int_set)
        

        

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        int_path = self.int_set_directory + self.int_set[idx]
        list_path = self.list_set_directory + self.list_set[idx]
        samplerate, x = wavfile.read(int_path)
        samplerate, y = wavfile.read(list_path)
        if(len(x) < 512):
            x = np.concatenate((x, np.zeros(512-len(x))))
        if(len(y) < 512):
            y = np.concatenate((y, np.zeros(512-len(y))))
        if(len(x) > 512):
            x = x[:512]
        if(len(y) > 512):
            y = y[:512]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        x = x[None,:]
        x = x.double()
        y = y[None,:]
        y = y.double()
        return (x,y)
    
    def getname(self,idx):
        return self.int_set[idx]





# Model class
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()

         
        self.enc_conv_1  = torch.nn.Conv1d(1 ,2 ,20, dtype=torch.float64, padding = "same")
        self.enc_conv_2  = torch.nn.Conv1d(2 ,4 ,20, dtype=torch.float64, padding = "same")
        self.enc_conv_3  = torch.nn.Conv1d(4 ,8 ,20, dtype=torch.float64, padding = "same")
        self.enc_conv_4  = torch.nn.Conv1d(8 ,16,20, dtype=torch.float64, padding = "same")
        self.bottom_conv = torch.nn.Conv1d(16,16,20, dtype=torch.float64, padding = "same")
        self.dec_fc_1    = torch.nn.Linear(1536,512, dtype=torch.float64)
        self.dec_fc_2    = torch.nn.Linear(1536,512, dtype=torch.float64)
        self.dec_fc_3    = torch.nn.Linear(1536,512, dtype=torch.float64)
        self.dec_fc_4    = torch.nn.Linear(1536,512, dtype=torch.float64)
        self.maxpool_2   = torch.nn.MaxPool1d(2)
        torch.nn.init.xavier_uniform(self.dec_fc_1.weight)
        torch.nn.init.xavier_uniform(self.dec_fc_2.weight)
        torch.nn.init.xavier_uniform(self.dec_fc_3.weight)
        torch.nn.init.xavier_uniform(self.dec_fc_4.weight)
    def forward(self,x):
        # x dims [512,1]
        layer1 = self.enc_conv_1(x)
        # [512,2]
        dummy = self.maxpool_2(layer1)
        # [256,2]
        layer2 = self.enc_conv_2(dummy)
        # [256,4]
        dummy = self.maxpool_2(layer2)
        # [128,4]
        layer3 = self.enc_conv_3(dummy)
        # [128,8]
        dummy = self.maxpool_2(layer3)
        # [64, 8]
        layer4 = self.enc_conv_4(dummy)
        # [64, 16]
        dummy = self.maxpool_2(layer4)
        # [32,16]
        dummy = self.bottom_conv(dummy)
        # [32,16]
        dummy = torch.flatten(dummy,start_dim=1)
        dummy = torch.cat((dummy,torch.flatten(layer4,start_dim=1)),dim=1)
        dummy = self.dec_fc_1(dummy)
        dummy = torch.cat((dummy,torch.flatten(layer3,start_dim=1)),dim=1)
        dummy = self.dec_fc_2(dummy)
        dummy = torch.cat((dummy,torch.flatten(layer2,start_dim=1)),dim=1)
        dummy = self.dec_fc_3(dummy)
        dummy = torch.cat((dummy,torch.flatten(layer1,start_dim=1)),dim=1)
        dummy = self.dec_fc_4(dummy)
        dummy = dummy[None,:]
        # print(dummy.size())
        

        # [256,1]
        return dummy
        







# DataLoader
training_set = Time_Dataset(test_folder)
test_set = Time_Dataset(test_folder)
validation_set = Time_Dataset(validation_folder)
train_loader = DataLoader(training_set, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# Load model
model = U_Net()

# Load state
if(load_params):
    model.load_state_dict(torch.load(model_filename))
    print("Params Loaded")

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
model.to(device)

# Loss Function
loss_fn = nn.MSELoss()

# Optimizer:

optimizer = torch.optim.Adam(model.parameters())

# from torchinfo import summary
# summary(model, input_size=(batch_size, 1, 28, 28), verbose=1, device=device);






iter_count = 0
for epoch in range(num_epochs):



    model.train()
    val_loss = 0
    for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1:02d}')):
        iter_count += 1

        # get inputs and expected outs
        inputs = data[0]
        expected = data[1]

        inputs = inputs.to(device)
        expected = expected.to(device)


        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs,expected)
        val_loss += loss
        

        loss.backward()
        optimizer.step()
    print("loss:",val_loss.item()/len(train_loader.dataset))


# for epoch in range(num_epochs):



#     model.train()
#     val_loss = 0
#     for i, data in enumerate(tqdm(validation_loader, desc=f'Epoch {epoch+ num_epochs + 1:02d}')):
#         iter_count += 1

#         # get inputs and expected outs
#         inputs = data[0]
#         expected = data[1]

#         inputs = inputs.to(device)
#         expected = expected.to(device)


#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = loss_fn(outputs,expected)
#         val_loss += loss
        

#         loss.backward()
#         optimizer.step()
#     print("loss:", loss.item()/len(validation_loader.dataset))
          
model.eval()


torch.save(model.state_dict(),model_filename)
print('Finished Training!')



def save_mat(exp_tens,out_tens,path,itr,name):
    exp_np_arr = exp_tens.detach().cpu().numpy()
    output_np_arr = out_tens.detach().cpu().numpy()
    for i in range(len(exp_np_arr)):
        in_dict = {"x": exp_np_arr[i][0], "label": "expected sample " + str(itr+i), "file": name}
        out_dict = {"x": output_np_arr[i][0], "label": "output sample " + str(itr+i), "file": name}
        scipy.io.savemat(path + "input " + str(itr+i) + ".mat", in_dict)
        scipy.io.savemat(path + "output " + str(itr+i) + ".mat", out_dict)


total_loss = 0
loss_itr = 0
model.eval()

for i, data in enumerate(test_loader):

    loss_itr += 1

    # get inputs and expected outs
    inputs = data[0]
    expected = data[1]

    inputs = inputs.to(device)
    expected = expected.to(device)



    outputs = model(inputs)

    save_mat(expected, outputs,result_folder,loss_itr,test_set.getname(loss_itr-1))
    

    total_loss += loss_fn(outputs,expected)

avg_loss = total_loss/loss_itr

print('Average Loss on Test Set:', avg_loss.item())


