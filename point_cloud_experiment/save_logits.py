import torch
import numpy as np
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.transforms import SamplePoints, KNNGraph
import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
from models_pnc import pointcloudNN, localnet, globalnet
import os
import random
from scipy.spatial.transform import Rotation

def filter_data(data):
    boole = (data.y in [0,3,4,5,6,7,9,10,11,13])
    return boole

def rots():
    random.seed(114)
    r = Rotation.random(100)
    return r

def save_logits(model,loader,device,rs,path,ood):

    G = np.size(rs)
    if ood:
        array = np.empty((520,101,10))
    else:
        array = np.empty((908,101,10))
    n=0
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader,desc='Batches completed:',leave=False):
            data = data.to(device)
            outputs = model(data)
            k=outputs.shape[0]

            array[n:n+k,0,:] = outputs.detach().cpu().numpy()

            pos = data.pos
            for g in range(G):
                data.pos = torch.t(torch.matmul(((torch.from_numpy(rs[g].as_matrix())).float()).to(device), torch.t(pos)))
                orbit_outputs = model(data)
                array[n:n+k,g+1,:] = orbit_outputs.detach().cpu().numpy()
            n=n+k
        np.save(path,array)
    return

PATH = os.getcwd() # Get current directory
PATH1 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_models/') # create new directory name
PATH2 = os.path.join(PATH, r'data/modelnet10/') # create new directory name
PATH3 = os.path.join(PATH, r'data/modelnet40/') # create new directory name
PATH4 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_logits/') # create new directory name
PATH5 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_logits_ood/') # create new directory name
if not os.path.isdir(PATH1): # if the directory does not already exist
    os.makedirs(PATH1) # make a new directory
else:
    pass
if not os.path.isdir(PATH2): # if the directory does not already exist
    os.mkdir(PATH2) # make a new directory
else:
    pass
if not os.path.isdir(PATH3): # if the directory does not already exist
    os.mkdir(PATH3) # make a new directory
else:
    pass
if not os.path.isdir(PATH4): # if the directory does not already exist
    os.mkdir(PATH4) # make a new directory
else:
    pass
if not os.path.isdir(PATH5): # if the directory does not already exist
    os.mkdir(PATH5) # make a new directory
else:
    pass
DATA_ROOT = './data/modelnet10/' # Root for loading dataset
DATA_ROOT2 = './data/modelnet40/' # Root for loading dataset
LOAD_ROOT = './ensemble_pointcloud_experiment/saved_models/' # Root for saving data
SAVE_PATH = './ensemble_pointcloud_experiment/saved_logits/' # Root for saving data
SAVE_PATH2 = './ensemble_pointcloud_experiment/saved_logits_ood/' # Root for saving data

BATCH_SIZE = 16

rs = rots()

test_data = datasets.ModelNet(
    root=DATA_ROOT,
    name='10',
    train=False,
    transform=pyg.transforms.Compose([SamplePoints(num=1024), KNNGraph(k=6)])
)

ood_test_data = datasets.ModelNet(
    root=DATA_ROOT,
    name='40',
    train=False,
    pre_filter=filter_data,
    transform=pyg.transforms.Compose([SamplePoints(num=1024), KNNGraph(k=6)])
)

testloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

ood_testloader = DataLoader(ood_test_data,batch_size=BATCH_SIZE,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for TASK_ID in range(1000):

    model = pointcloudNN(3,100,10,localnet,globalnet)

    model.to(device)
    if device.type == 'cuda':
        model.load_state_dict(torch.load(os.path.join(LOAD_ROOT,'PCNN'+'_'+str(TASK_ID)+'_'+'epoch'+str(10)), weights_only=True))
    else:
        model.load_state_dict(torch.load(os.path.join(LOAD_ROOT,'PCNN'+'_'+str(TASK_ID)+'_'+'epoch'+str(10)), weights_only=True, map_location=torch.device('cpu')))

    save_logits(model=model, loader=testloader, device=device, rs=rs, path=os.path.join(SAVE_PATH, 'logits'+'_'+'model'+'_'+str(TASK_ID)),ood=False)
    save_logits(model=model, loader=ood_testloader, device=device, rs=rs, path=os.path.join(SAVE_PATH2, 'logits'+'_'+'model'+'_'+str(TASK_ID)),ood=True)






