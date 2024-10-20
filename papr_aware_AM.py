# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:32:18 2024

@author: nickm
"""

import torch 
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fft import ifft
from torch.utils.data import DataLoader, TensorDataset

#creeating the network
from torch import nn
class net(nn.Module):
  def __init__(self,input_size,output_size):
    super(net,self).__init__()
    self.l1 = nn.Linear(input_size, 100)
    self.tanh = nn.Tanh() 
    self.l4 = nn.Linear(100,100)
    self.l2 = nn.Linear(100,output_size)
    self.l3 = nn.Softmax()
    self.drop = nn.Dropout(p=0)
    # self.gumbel = (logits, hard= True, dim = -1)
  def forward(self,x): 
    output = self.l1(x) 
    output = self.tanh(output) 
    output = self.l4(output)
    output = self.tanh(output) 
    output = self.l2(output)

    y1 = self.l3(output[:,0:K_d])
    
    # use of two different activation functions in the last layer
    # the for loop can be vectorized for efficiency
    # the output is K_d x M 
    for i in range(K_d):        
        y2 = nn.functional.gumbel_softmax(output[:,K_d+i*J:(i+1)*J+K_d], hard= True, dim = -1)
        y1 = torch.cat((y1, y2), -1)
    output = y1
    # print(output.size())
    return output


def loss_function_vol2(y,order,K_d,h_k, N_0,epsilon, Pmax):
   term1 = torch.FloatTensor([0])
   
   # this for loop produces the throughput log_2(M)*(1-P_s), P_s: SER, M: modulation order
   for i in range(K_d): 
       # gamma: SNR
       gamma = Pmax*y[0,i]*h_k[0,i]/(N_0)
       ser = 1 - (1 - 2*(torch.sqrt(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order))-1)/(torch.sqrt(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order)))*0.5*torch.erfc(1/1.4142*torch.sqrt(3*gamma/(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order)-1))))**2
       term1 = term1 + torch.log(torch.log2(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order))*(1-ser))
       #torch.log(torch.log2(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order))
   
   # code below is a small monte carlo procedure to output the average PAPR based on the 
   # modulation selection of the DNN. Then, the average consumption of the PA is given.
   av_papr = 0
   symbol = torch.zeros((1,K_d))
   symbol  = symbol.to(torch.complex32)

   vector = torch.zeros((K_d,1))
   
   # l is a parameter which affects how many 0s will be padded to the time domain signal
   # to increase the accuracy of the DFT
   l=3
   counter=1*10**2 #300
   ap_av=0
   vector_tot = 0
   for j in range(counter):
        symbol = torch.zeros((1,K_d))
        symbol  = symbol.to(torch.complex32)
        for i in range (K_d):
              # index = np.where(a[:,i]==1)
              # index=int(index[0]) 
              D=2**(0.5*torch.log2(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order)))
              D = int(D.detach().numpy())
              Delta= torch.sqrt(2/3*(torch.sum(y[:,K_d+i*J:(i+1)*J+K_d]*order)-1))
              Delta = Delta.detach().numpy()
              # psi=torch.random.randint(1,2*D+1)
              psi=torch.randint(1, 2*D+1, (1,))
              z=torch.randint(1, 2*D+1, (1,))
              imag=(2*z-2*D-1)/Delta
              real = (2*psi-2*D-1)/Delta 
              symbol[0,i] = complex(real, imag)

        symbol = symbol*torch.sqrt(y[:,0:K_d])
        symbol=torch.cat((symbol[:,0:int(K_d/2)], torch.zeros(1,l*K_d), symbol[:,int(K_d/2):K_d]), dim=-1)
        ofdma_symbol = 1*torch.fft.ifft(symbol)  

        vector = torch.abs(ofdma_symbol)**2
        aa = torch.mean(vector)
        papr = torch.max(vector)/aa
        ap_eff = 0.5/papr
        ap_av = ap_av+ap_eff
        av_papr = av_papr+papr
        vector_tot += vector            
   av_papr = av_papr/counter
   ap_aver=ap_av/counter
   vector_tot = vector_tot/counter
   consumption = torch.sum(y[0,0:K_d])**epsilon*Pmax**(1-epsilon)/(ap_aver)
   
   # averaga energy efficiency of the selected modulation-power per subcarrier
   x =  -term1/(1*consumption)
   return x 

def prepare_dataloader(size_tot, K_d, batch_size, device):

    # utils.plot_constellation(qam_symbols, "qam_constellation.png")
    htot = np.random.exponential(scale=1, size=(size_tot, K_d))
    h = torch.from_numpy(htot)
    h = h.to(torch.float32)
    
    inputs = h
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    dataset = TensorDataset(inputs_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

######################### MAIN CODE BEGGINING #########################


K_d = 64
order = torch.tensor([[4, 64, 128, 256, 1024]])  # Modulation orders
J = order.size(dim=1)

# Noise power and other constants
Pmax = 10
B = 5 * 10**6
N_0 = 10**(-135/10) / (K_d * B * Pmax)
epsilon = 0.5

# Hyperparameters
batch_size = 64
epochs = 151
learning_rate = 0.005
size_tot = 5000  # Total number of samples

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare DataLoader
train_loader = prepare_dataloader(size_tot, K_d, batch_size, device)

# Model Initialization
model = net(K_d, (1 + J) * K_d).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize variable for tracking loss over epochs
var = torch.zeros(1, epochs)

# Training Loop for unsupervised learning -- online learning uses the same code  
# but the DNN is fully trained for each different instance of h.
for epoch in range(epochs):
    for batch_idx, (h_batch,) in enumerate(train_loader):
        h_batch = h_batch.to(device)  # Ensure the batch is on the correct device

        # Forward pass: compute predictions
        y_pred = model(h_batch)
        
        # Split output into power (pk) and modulation (arr)
        pk = y_pred[:, :K_d]
        arr = y_pred[:, K_d:].reshape(J, K_d)
        
        # Initialize at first epoch
        if epoch == 0 and batch_idx == 0:
            ptot = pk
            at = arr
        else:
            ptot = torch.cat((ptot, pk), 0)
            at = torch.cat((at, arr), 0)
        
        # Compute cost using the custom loss function
        cost = loss_function_vol2(y_pred, order, K_d, h_batch, N_0, epsilon, Pmax)

        # Accumulate the cost for the current epoch
        var[:, epoch] += cost.item()

        # Zero gradients, perform backpropagation, and update weights
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Print progress every 5 epochs
        if epoch % 5 == 0 and batch_idx == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {cost.item()}")
      
ptot = ptot.detach().numpy()
at = at.detach().numpy()
# var = var/10
var = var.detach().numpy()