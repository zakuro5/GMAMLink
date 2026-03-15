import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np

class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)


    def Adj_Generate(self,TF_set,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)


        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()
        return adj

class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(GCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def norm_adj(self, adj):

        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)

        return adj_normalized

    def forward(self, x, adj):
        adj = self.norm_adj(adj)

        h = torch.matmul(x, self.weight)
        h = torch.matmul(adj, h)

        if self.bias is not None:
            h = h + self.bias

        return F.relu(h)


        if self.bias is not None:
            h = h + self.bias

        return F.relu(h)
