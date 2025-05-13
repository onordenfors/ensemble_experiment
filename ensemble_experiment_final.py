import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import torchvision.transforms as transforms
import networks_final
from trainer_final import train
from tqdm import trange
from augmentation_final import randomC4, randomC16

def experiment(sym,syminit,cnn,members, epochs, batch=32, lr=0.01, wd=0,group = 'C4'):
    PATH = os.getcwd() # Get current directory
    PATH1 = os.path.join(PATH, 'EnsembleExperiment',group,'Members') # create new directory name
    PATH2 = os.path.join(PATH, r'data/') # create new directory name
    if not os.path.isdir(PATH1): # if the directory does not already exist
        os.makedirs(PATH1) # make a new directory
    else:
        pass
    if not os.path.isdir(PATH2): # if the directory does not already exist
        os.mkdir(PATH2) # make a new directory
    else:
        pass
    LOAD_ROOT = './data/' # Root for loading dataset
    SAVE_ROOT = os.path.join('EnsembleExperiment',group,'Members') # Root for saving data

    TASKS = members # Total number of tasks/members across architectures

    # Set number of epochs
    EPOCHS = epochs

    # Set batch size for training
    BATCH_SIZE = batch

    # Load MNIST and calculate mean and standard deviation
    train_data = datasets.MNIST(
        root=LOAD_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    muhat = train_data.data.float().mean()/255
    sigmahat = train_data.data.float().std()/255

    # Create transforms for the data, including normalization w.r.t. mean and standard deviation,
    # and a random rotation of K*pi/2 radians, K integer
    if group == 'C4':
        train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(muhat,sigmahat),randomC4])
    elif group == 'C16':
        train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(muhat,sigmahat),randomC16])

    # Load data with transforms applied
    train_data = datasets.MNIST(
        root=LOAD_ROOT,
        train=True,
        download=True,
        transform=train_transform
    )

    # Create dataloader for training data
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Give device as 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Give criterion as cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    # Set learning rate
    LEARNING_RATE = lr

    # Set weight decay
    WEIGHT_DECAY = wd

    # Select architecture
    omega=torch.zeros(3,3)
    omega[0,1]=1
    omega[1,0]=1
    omega[1,2]=1
    omega[2,1]=1
    '''omega=torch.zeros(5,5)
    omega[0,2]=1
    omega[1,2]=1
    omega[2,0]=1
    omega[2,1]=1
    omega[2,2]=1
    omega[2,3]=1
    omega[2,4]=1
    omega[3,2]=1
    omega[4,2]=1'''
    if sym:
        omega[1,1]=1
        '''omega[1,1]=1
        omega[1,3]=1
        omega[3,1]=1
        omega[3,3]=1'''
        FILE_NAME = 'SYMM'
    else:
        omega[0,0]=1
        '''omega[0,0]=1
        omega[0,1]=1
        omega[1,0]=1
        omega[1,1]=1'''
        FILE_NAME = 'ASYM'
    if syminit:
        INV='SYMINIT'
    else:
        INV='ASYMINIT'

    if cnn:
        FILE_NAME ='CNN'
        INV = ''
        
    # train ensemble members
    for TASK_ID_INT in trange(TASKS,desc='Tasks completed:',leave=False):
        if cnn:
             model = networks_final.CNN()
        elif sym:
            INV='SYMINIT'
            model = networks_final.omegaCNN(omega)
        else:
            if syminit:
                model = networks_final.omegaCNNAsym(omega)
            else:
                model = networks_final.omegaCNN(omega)
           

        # Initialize the chosen model
        model.init_weights()
        model = model.to(device)

        # Set optimizer as SGD
        optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

        # Save the untrained model
        torch.save(model.state_dict(), os.path.join(SAVE_ROOT,FILE_NAME+'_'+INV+'_'+str(TASK_ID_INT)+'_'+str(0)))


        for epoch in trange(EPOCHS, desc='Epochs completed:', leave=False):
            # Train for a number of epochs and save model dictionary at the end of each epoch

            train(model,trainloader,optimizer,criterion,device)
            completed_epochs = epoch + 1
            torch.save(model.state_dict(), os.path.join(SAVE_ROOT,FILE_NAME+'_'+INV+'_'+str(TASK_ID_INT)+'_'+str(completed_epochs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that trains ensemble members"
    )
    parser.add_argument("--sym", required=False, dest = 'symm', action='store_true')
    parser.add_argument("--asym", required=False, dest = 'symm', action='store_false')
    parser.add_argument("--cnn", required=False, dest = 'cnn', action='store_true')
    parser.add_argument("--syminit", required=False, dest = 'symminit', action='store_true')
    parser.add_argument("--asyminit", required=False, dest = 'symminit', action='store_false')
    parser.add_argument("--members", required=False, type=int, default=1000)
    parser.add_argument("--epochs", required=False, type=int, default=10)
    parser.add_argument("--C4", required=False, dest = 'group4', action='store_true')
    parser.add_argument("--C16", required=False, dest = 'group4', action='store_false')
    
    args = parser.parse_args()

    sym = args.symm
    syminit = args.symminit
    members = args.members
    epochs = args.epochs
    group4 = args.group4
    cnn = args.cnn

    if group4:
        group = 'C4'
    else:
        group = 'C16'
    experiment(sym,syminit,cnn,members,epochs,batch=32,lr=0.01,wd=0,group=group)