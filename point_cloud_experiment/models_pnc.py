import math
import numpy as np
import torch
from torch_geometric.nn import PointNetConv
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, nin, nhid, nout):
        super().__init__()
        self.lin1 = nn.Linear(nin, nhid[0])
        self.lin2 = nn.Linear(nhid[0], nhid[1])
        self.lin3 = nn.Linear(nhid[1], nout)
        self.activate = nn.Tanh()
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.lin3.weight, gain=5/3)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activate(x)
        x = self.lin2(x)
        x = self.activate(x)
        x = self.lin3(x)
        return x

class localnet(nn.Module):
    # local net for use in pointNet

    def __init__(self, nhid, nout):
        super().__init__()
        self.nhid = nhid
        self.nout = nout
        self.lin1 = nn.Linear(nhid+3, nhid+3)
        self.lin2 = nn.Linear(nhid+3, nout)
        self.activate = nn.Tanh()
        self.bn1 = nn.LayerNorm(normalized_shape=(7,nhid+3),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(7,nout),elementwise_affine=False)
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=5/3)

    def forward(self, x):
        #print('Before',x.shape)
        x = self.lin1(x)
        #print('First', x.shape)
        x = self.activate(x)
        x = x.view(int(x.size(0)/7),7,-1)
        x = self.bn1(x)
        x = x.view(-1,self.nhid+3)
        x = self.lin2(x)
        #print('Second', x.shape)
        x = self.activate(x)
        x = x.view(int(x.size(0)/7),7,-1)
        x = self.bn2(x)
        x = x.view(-1,self.nout)
        return x

class globalnet(nn.Module):
    # global net for use in pointNet

    def __init__(self, nhid, nout):
        super().__init__()
        self.nhid = nhid
        self.nout = nout
        self.lin1 = nn.Linear(nhid, nhid)
        self.lin2 = nn.Linear(nhid, nout)
        self.bn1 = nn.LayerNorm(normalized_shape=(nhid),elementwise_affine=False)
        self.bn2 = nn.LayerNorm(normalized_shape=(nout),elementwise_affine=False)
        self.activate = nn.Tanh()
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=5/3)

    def forward(self, x):
        x = self.lin1(x)
        x = self.activate(x)
        x = self.bn1(x)
        x = self.lin2(x)
        x = self.activate(x)
        x = self.bn2(x)
        return x

class pointcloudNN(nn.Module):
    # graph NN using pointNet layer

    def __init__(self, nfeat, nhid, nclass, localNN, globalNN):
        super().__init__()
        self.enc1 = nn.Linear(nfeat,int(nhid/2))
        self.enc2 = nn.Linear(int(nhid/2),int(nhid/2))
        self.enc3 = nn.Linear(int(nhid/2),nhid)
        self.pnclayer = PointNetConv(local_nn=localNN(nhid,2*nhid),global_nn=globalNN(2*nhid,3*nhid),aggr='mean')
        #self.pnclayer2 = PointNetConv(local_nn=localNN(nhid,2*nhid),global_nn=globalNN(2*nhid,nhid))
        #self.dec = nn.Linear(3*nhid, nclass)
        self.dec1 = nn.Linear(3*nhid,2*nhid) 
        self.dec2 = nn.Linear(2*nhid, nhid)
        self.dec3 = nn.Linear(nhid, nclass)
        self.activate = nn.Tanh()
        torch.nn.init.xavier_normal_(self.dec1.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.dec2.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.dec3.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.enc1.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.enc2.weight, gain=5/3)
        torch.nn.init.xavier_normal_(self.enc3.weight, gain=5/3)


    def forward(self, data):
        x = data.pos.float()
        pos = data.pos
        edge_index = data.edge_index
        x = self.enc1(x)
        x = self.activate(x)
        x = self.enc2(x)
        x = self.activate(x)
        x = self.enc3(x)
        x = self.activate(x)
        x = self.pnclayer(x=x,pos=pos,edge_index=edge_index)
        #x = self.activate(x)
        #x = self.pnclayer2(x=x,pos=pos,edge_index=edge_index)
        #x = self.activate(x)
        x = x.view(int(x.size(0)/1024),1024,-1)
        #x = x.view(int(x.size(0)/1024),-1)
        x = x.mean(1)
        x = self.dec1(x)
        x = self.activate(x)
        x = self.dec2(x)
        x = self.activate(x)
        x = self.dec3(x)
        return x
    

class Ensemble(nn.Module):
    def __init__(self,members):
        super().__init__()
        self.members = members

    def forward(self,x):
        sum=0
        n = 0
        for i, member in enumerate(self.members):
            sum = sum + member(x)
            n = n + 1
        x = sum/n
        return x