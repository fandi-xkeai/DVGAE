import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from utils import *
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)

# This implementation is modified based on:  https://github.com/SsGood/ADGCN/blob/main/network.py.

class Discriminator(nn.Module):
    def __init__(self, nfeatures, ncaps):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(nfeatures, nfeatures//2)
        self.cls1 = nn.Linear(nfeatures//2, 1)
        self.cls2 = nn.Linear(nfeatures//2, ncaps)
    def forward(self,x):
        x = fn.relu(self.linear(x))
        logits = self.cls1(x)
        y_ = self.cls2(x)
        return logits, y_

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight).to(device))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias).to(device))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight.to(device), bias.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x): 
        return torch.mm(x, self.weight).to(device) + self.bias.to(device)
    
    
class RoutingLayer(nn.Module):
    def __init__(self, dim=1, num_caps=1):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d).to(device)
        self._cache_zero_k = torch.zeros(1, self.k).to(device)

    def forward(self, x, neighbors, max_iter, param):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
            
        n, m = x.size(0), neighbors.size(1)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0).to(device)
        z = z[neighbors].view(n, m, k, delta_d)
        u = x.view(n, k, delta_d)
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p1 = fn.softmax(p, dim=1)
            u1 = torch.sum(z * p1.view(n, m, k, 1), dim=1).to(device)
            
            p2 = fn.softmax(p, dim=2)
            u2 = torch.sum(z * p2.view(n, m, k, 1), dim=1).to(device)
            
            u = param * u1 + (1-param) * u2 
            
            u = u + x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)

    
class CapsuleNet(nn.Module):  
    def __init__(self, nfeat, hyperpm):
        super(CapsuleNet, self).__init__()
        self.nhidden = hyperpm.nhidden
        self.ncaps  = hyperpm.ncaps
        rep_dim = self.nhidden * self.ncaps
        self.nlayer = hyperpm.nlayer
        self.pca1 = SparseInputLinear(nfeat, rep_dim)
        
        self.param = nn.Parameter(torch.ones(1))
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, self.ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
       
        
        self.conv_ls = conv_ls
        
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit
        self.discriminator = Discriminator(self.nhidden, self.ncaps)
        

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        dev = x.device
        x1 = fn.leaky_relu(self.pca1(x)).to(device)#mean

        x_initial1 = x1

        param = torch.sigmoid(self.param).to(device)
        for i, conv in enumerate(self.conv_ls):
                x1 = self._dropout((conv(x1, nb, self.routit, param)))
                if i == self.nlayer-2:
                    break
        x1 = conv(x1, nb, self.routit, param).to(device)

              
        return x_initial1,  x1

