import os
import torch
import torch_geometric as pyg
from torch_geometric.transforms import SamplePoints, KNNGraph
import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
from so3haar import haarsample
from models_pnc import pointcloudNN, localnet, globalnet
from trainer_pnc import train

PATH = os.getcwd() # Get current directory
PATH1 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_models/') # create new directory name
PATH2 = os.path.join(PATH, r'data/modelnet10/') # create new directory name
if not os.path.isdir(PATH1): # if the directory does not already exist
    os.makedirs(PATH1) # make a new directory
else:
    pass
if not os.path.isdir(PATH2): # if the directory does not already exist
    os.mkdir(PATH2) # make a new directory
else:
    pass
LOAD_ROOT = './data/modelnet10/' # Root for loading dataset
SAVE_ROOT = './ensemble_pointcloud_experiment/saved_models/' # Root for saving data

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 0

# Load ModelNet and create point cloud
train_data = datasets.ModelNet(
    root=LOAD_ROOT,
    name='10',
    train=True,
    transform=pyg.transforms.Compose([SamplePoints(num=1024), KNNGraph(k=6), haarsample])
)

# Create dataloader for training data
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Give device as 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Give criterion as cross entropy
criterion = torch.nn.CrossEntropyLoss()

# Load model
model = pointcloudNN(3, 100, 10, localnet, globalnet)
model = model.to(device)

# Set optimizer as SGD
optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)


num_models = 1000
for TASK_ID in range(num_models):

    TASK_ID = str(TASK_ID)
    completed_epochs = 0
    torch.save(model.state_dict(), os.path.join(SAVE_ROOT,'PCNN'+'_'+TASK_ID+'_'+'epoch'+str(completed_epochs)))
    for epoch in range(EPOCHS):
    
        train(model,trainloader,optimizer,criterion,device)
        completed_epochs = epoch + 1
        torch.save(model.state_dict(), os.path.join(SAVE_ROOT,'PCNN'+'_'+TASK_ID+'_'+'epoch'+str(completed_epochs)))