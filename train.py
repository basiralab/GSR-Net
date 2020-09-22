import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *

criterion = nn.MSELoss()

def train(model, optimizer, subjects_adj,subjects_labels, args):
  
  i = 0
  all_epochs_loss = []
  no_epochs = args.epochs

  for epoch in range(no_epochs):

      epoch_loss = []
      epoch_error = []

      for lr,hr in zip(subjects_adj,subjects_labels):

          model.train()
          optimizer.zero_grad()
          
          lr = torch.from_numpy(lr).type(torch.FloatTensor)
          hr = torch.from_numpy(hr).type(torch.FloatTensor)
          
          model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
          model_outputs  = unpad(model_outputs, args.padding)

          padded_hr = pad_HR_adj(hr,args.padding)
          eig_val_hr, U_hr = torch.symeig(padded_hr, eigenvectors=True,upper=True)
          
          loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights,U_hr) + criterion(model_outputs, hr) 
          
          error = criterion(model_outputs, hr)
          
          loss.backward()
          optimizer.step()

          epoch_loss.append(loss.item())
          epoch_error.append(error.item())
      
      i+=1
      print("Epoch: ",i, "Loss: ", np.mean(epoch_loss), "Error: ", np.mean(epoch_error)*100,"%")
      all_epochs_loss.append(np.mean(epoch_loss))

#   plt.plot(all_epochs_loss)
#   plt.title('GSR-UNet with self reconstruction: Loss')
#   plt.show(block=False)
    
def test(model, test_adj, test_labels,args):

  test_error = []
  preds_list=[]
  g_t = []
  
  i=0
  # TESTING
  for lr, hr in zip(test_adj,test_labels):

    all_zeros_lr = not np.any(lr)
    all_zeros_hr = not np.any(hr)

    if all_zeros_lr == False and all_zeros_hr==False: #choose representative subject
      lr = torch.from_numpy(lr).type(torch.FloatTensor)
      np.fill_diagonal(hr,1)
      hr = torch.from_numpy(hr).type(torch.FloatTensor)
      preds,a,b,c = model(lr)
      preds = unpad(preds, args.padding)

      #plot residuals
    #   if i==0:
    #     print ("Hr", hr)     
    #     print("Preds  ", preds)
    #     plt.imshow(hr, origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
    #     plt.show(block=False)
    #     plt.imshow(preds.detach(), origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
    #     plt.show(block=False)
    #     plt.imshow(hr - preds.detach(), origin = 'upper',  extent = [-0.5, 268-0.5, 268-0.5, -0.5])
    #     plt.show(block=False)
      
      preds_list.append(preds.flatten().detach().numpy())
      
      error = criterion(preds, hr)
      g_t.append(hr.flatten())
      print(error.item())
      test_error.append(error.item())
     
      i+=1

  print ("Test error MSE: ", np.mean(test_error))
  
  #plot histograms
#   preds_list = [val for sublist in preds_list for val in sublist]
#   g_t_list = [val for sublist in g_t for val in sublist]
#   binwidth = 0.01
#   bins=np.arange(0, 1 + binwidth, binwidth)
#   plt.hist(preds_list, bins =bins,range=(0,1),alpha=0.5,rwidth=0.9, label='predictions')
#   plt.hist(g_t_list, bins=bins,range=(0,1),alpha=0.5,rwidth=0.9, label='ground truth')
#   plt.xlim(xmin=0, xmax = 1)
#   plt.legend(loc='upper right')
#   plt.title('GSR-Net with self reconstruction: Histogram')
#   plt.show(block=False)


