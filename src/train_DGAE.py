import argparse
import os
import time
import dgl
import model_DGAE
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from input_data import DataReader
from preprocess import (
    mask_test_edges,
    preprocess_graph,
    sparse_to_tuple,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from dgl import AddSelfLoop
import pickle
import random
import sys
import tempfile
import gc
import matplotlib.cm
import scipy.sparse as spsprs
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl

'''
Our code is based on: https://github.com/dmlc/dgl/blob/master/examples/pytorch/vgae/train.py. 
Thank them for their code!
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser(description="Disentangled Graph Auto-Encoder")
parser.add_argument('--datadir', type=str, default='./data/')
parser.add_argument('--datname', type=str, default='cora')
parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument("--epochs", "-e", type=int, default=3000, help="Number of epochs to train.")
parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
parser.add_argument('--ncaps', type=int, default=2,
                        help='Maximum number of capsules per layer.')
parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
parser.add_argument('--routit', type=int, default=5,
                        help='Number of iterations when routing.')
parser.add_argument('--nbsz', type=int, default=20,
                        help='Size of the sampled neighborhood.')
parser.add_argument('--test', type=str, default=True,
                        help='val_test flag')
parser.add_argument('--seed', type=int, default=23,
                        help='random seed')
parser.add_argument('--cls_layer', type=int, default=2,
                        help='number of layers in classifier. Must be larger than 0')
parser.add_argument('--nhid', type=int, default=128)#intermediate feature dimension
parser.add_argument('--weight', type=float, default=0.0001)
parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n epochs")
parser.add_argument('--resume', action='store_true', default=False, help='whether load model from checkpoint')#DEAR
parser.add_argument('--encoder', type=str, default='disen', help="encoder type only for name")
parser.add_argument('--decoder', type=str, default='factorProduct', help="decoder type only for name")
parser.add_argument('--permute', type=str, default='swap', help="permute method in flow")
parser.add_argument('--features', default=False, type=bool,  help="whether use features")
args = parser.parse_args()
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)


def compute_loss_para(adj):#与原代码一致
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

    
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   
    
    
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': 
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming': 
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal': 
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0) 
    print('initialize network with %s type' % init_type)
    net.apply(init_func)   
    

def main():
    torch.autograd.set_detect_anomaly = True
    test = args.test
    test_seeds = 2009332812
    if test:
        seeds = test_seeds

    
    set_rng_seed(seeds)
    datname = args.datname.split('_', 1)[0].replace('.', '')
    dataset = DataReader(datname,args.datadir, seeds, test)
  
         
    graph, features, targ = dataset.get_graph_feat_targ()
    adj1, adj = dataset.get_graph_adj()
    features = sparse_to_tuple(features.tocoo())
    adj_orig = adj
    
    adj_orig = adj_orig - sp.dia_matrix(
        (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape
    )
    adj_orig.eliminate_zeros()
    (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_normalization, adj_norm = preprocess_graph(adj)

    # Create model
    graph = dgl.from_scipy(adj_normalization)

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    ).to(device)
    adj_label = torch.sparse.FloatTensor(
        torch.LongTensor(adj_label[0].T),
        torch.FloatTensor(adj_label[1]),
        torch.Size(adj_label[2]),
    ).to(device)
    features = torch.sparse.FloatTensor(
        torch.LongTensor(features[0].T),
        torch.FloatTensor(features[1]),
        torch.Size(features[2]),
    ).to(device)

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight

    features = features.to_dense()
    in_dim = features.shape[-1]
    if not args.features:
        features = torch.eye(features.size(0)).to(device)
    print(features)
    vgae_model = model_DGA.VGAEModel(dataset,features, args)
    
    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    )

    def make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
        
    def write_config_to_file(config, save_path):
        with open(os.path.join(save_path, 'config.txt'), 'w') as file:
            for arg in vars(config):
                file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')
                
    save_dir = './results/{}/{}_{}_{}_weight_{}_seed_{}/'.format(
        args.datname, args.encoder, args.decoder,args.permute, str(args.weight),str(test_seeds))
    make_folder(save_dir)
    write_config_to_file(args, save_dir)
    
    log_file_name = os.path.join(save_dir, 'log.txt')
    global log_file
    if args.resume:
        log_file = open(log_file_name, "at")
    else:
        log_file = open(log_file_name, "wt")
        
        
    def get_scores(edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy
    
    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        vgae_model.train()

        z, logits = vgae_model.forward()
        
        loss = F.binary_cross_entropy(
            logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
   
        loss += vgae_model.compute_disentangle_loss(args.weight)
        
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj_label)
        
        vgae_model.eval()
        
        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits.cpu())

        # Print out performance
        log = ('Train Epoch: {} \t train_loss: {:.5f}, cross_entropy: {:.5f},  '
                   'disentangle_loss: {:.5f}, train_acc: {:.5f}, val_roc: {:.5f}, val_ap: {:.5f}, time: {:.5f}'.format(
                epoch+1, loss.item(),
                F.binary_cross_entropy(
            logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
                       , vgae_model.compute_disentangle_loss(args.weight).item(),
                train_acc, val_roc,val_ap,time.time() - t ))
        print(log)
        log_file.write(log + '\n')
        log_file.flush() 
        
            
    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits.cpu())

    log = ('End of training! test_roc: {:.5f},  test_ap: {:.5f}'.format(test_roc, test_ap))
    print(log)
    log_file.write(log + '\n')
    log_file.flush()   

if __name__ == "__main__":
        main()
