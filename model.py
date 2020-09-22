import torch
import torch.nn as nn
from layers import *
from ops import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F

class GSRNet(nn.Module):

  def __init__(self,ks,args):
    super(GSRNet, self).__init__()
    
    self.lr_dim = args.lr_dim
    self.hr_dim = args.hr_dim
    self.hidden_dim = args.hidden_dim
    self.layer = GSRLayer(self.hr_dim)
    self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
    self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu)
    self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu)

  def forward(self,lr):

    I = torch.eye(self.lr_dim).type(torch.FloatTensor)
    A = normalize_adj_torch(lr).type(torch.FloatTensor)

    self.net_outs, self.start_gcn_outs = self.net(A, I)
    
    self.outputs, self.Z = self.layer(A, self.net_outs)
    
    self.hidden1 = self.gc1(self.Z, self.outputs)
    self.hidden2 = self.gc2(self.hidden1, self.outputs)

    z = self.hidden2
    z = (z + z.t())/2
    idx = torch.eye(self.hr_dim, dtype=bool) 
    z[idx]=1
    
    return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs