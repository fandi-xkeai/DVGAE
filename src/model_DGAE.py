import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from network_DGAE import CapsuleNet
import numpy as np
import scipy.sparse as spsprs
import networkx as nx
import torch.autograd
import torch.optim as optim
from utils import *
from torch.nn.parameter import Parameter

import normflows as nf
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)
class MLP(nn.Module):
    def __init__(self, in_feat, hidden_size, out_size, layers=2, dropout=0.1):
        super(MLP, self).__init__()

        modules = []
        in_size = in_feat
        for layer in range(layers-1):
            modules.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
            modules.append(nn.LeakyReLU(0.1))
        modules.append(nn.Linear(in_size, out_size))

        self.model = nn.Sequential(*modules)

    def forward(self, features, cls=False):
        output = self.model(features)
        if cls:
            return F.log_softmax(output, dim=1)
        else:
            return output
        
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all

class VGAEModel(nn.Module):
    def __init__(self, dataset,features, hyperpm):
        super(VGAEModel, self).__init__()
        graph, feats, targ = dataset.get_graph_feat_targ()
        self.graph, self.feats = graph, features
        self.nbsz = hyperpm.nbsz
        self.ncaps = hyperpm.ncaps
        self.nhidden = hyperpm.nhidden
        
        nfeat = features.size(1)
        disen_model = CapsuleNet(nfeat, hyperpm).to(device)
        
        self.nfeat=nfeat
        self.disen_model=disen_model
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(device)
        
        self.classifier1 = MLP(in_feat=hyperpm.nhidden, hidden_size=hyperpm.nhid, out_size=hyperpm.ncaps, layers=hyperpm.cls_layer)
        self.classifier1.to(device)
        self.criterion = nn.NLLLoss()
    
        self.rk_lgt = Parameter(torch.FloatTensor(torch.Size([1, self.ncaps*self.nhidden])))
        self.reset_parameters()
              
    def encoder(self,features):
        self.x_initial1,self.mean = self.disen_model(features, self.neib_sampler.sample())
        
        return self.mean

    def factordecoder(self, z):
        dim_z = self.ncaps*self.nhidden
        assert dim_z == z.size(1)
        adj_tmp = torch.zeros([self.ncaps,z.size(0),z.size(0)])
        disen_feat = z.view(z.size(0), self.ncaps, self.nhidden)
        disen_feat = F.normalize(disen_feat, dim = 2)
        temp = disen_feat.unsqueeze(dim=1) * disen_feat
        temp = temp.sum(dim=-1).max(dim = -1).values
        adj_rec_final = torch.sigmoid(torch.matmul(z, z.t()) + temp)
        return adj_rec_final

    def InnerProductdecoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec
    
    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self):
        z = self.encoder(self.feats)
        adj_rec = self.factordecoder(z)
        return z, adj_rec

    def compute_disentangle_loss(self, loss_weight): 
        projection_mean = torch.reshape(self.x_initial1, (self.x_initial1.size(0), self.ncaps, self.nhidden)).to(device)     
        projection_mean1 = torch.reshape(projection_mean,(projection_mean.size(0)*self.ncaps, self.nhidden)).to(device)

        label = torch.tensor([i for i in range(self.ncaps)]).to(device)
        labels = label.repeat(self.mean.size(0))
        labels = labels.to(device)

        pred_label_projection_mean = self.classifier1(projection_mean1,cls=True)
        loss_projection_mean = self.criterion(pred_label_projection_mean, labels)

        loss =  loss_projection_mean 
        loss = loss * loss_weight
        
        return loss.to(device)


