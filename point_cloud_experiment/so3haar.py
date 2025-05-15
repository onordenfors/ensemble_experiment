import torch
import numpy as np
import torch_geometric
from scipy.spatial.transform import Rotation

def rots():
    r = Rotation.random(100)
    return r

def haarsample(data):
    R = Rotation.random()
    M = R.as_matrix()
    T = torch.from_numpy(M)
    T = T.float()
    tpose = torch.t(data.pos)
    data.pos = torch.t(torch.matmul(T,tpose))
    return data

def haarsample_v2(data):
    R = Rotation.random()
    M = R.as_matrix()
    T = torch.from_numpy(M)
    T = T.float()
    data.pos = torch.matmul(data.pos,T)
    return data

def mygaussian(data):
    data.pos = 10*torch.randn((1024,3))
    return data