import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import networks_final
from equivariance_tester_final import test
from generate_subensembles_final import subensembles
from tqdm import tqdm, trange



def evaluate(sym,syminit,member,epoch,indistribution):
    PATH = os.getcwd() # Get current directory
    PATH1 = os.path.join(PATH, r'EnsembleExperiment/Experiment/') # create new directory name
    PATH2 = os.path.join(PATH, r'data/') # create new directory name
    PATH3 = os.path.join(PATH, r'EnsembleExperiment/Subensembles/')
    PATH4 = os.path.join(PATH, r'EnsembleExperiment/Evaluation/')
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

    LOAD_ROOT = './data/' # Root for loading dataset
    LOAD_ROOT2 = './EnsembleExperiment/Members/'
    LOAD_ROOT3 = './EnsembleExperiment/Subensembles/'
    SAVE_ROOT = './EnsembleExperiment/Evaluation/' # Root for saving data

    # Set group size
    G = 4

    # Set batch size
    BATCH_SIZE = 100

    # Give device as 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Give criterion as cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    # Select data

    if indistribution==0:

        # Load MNIST and calculate mean and standard deviation
        train_data = datasets.MNIST(
            root=LOAD_ROOT,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        muhat = train_data.data.float().mean()/255
        sigmahat = train_data.data.float().std()/255

        # Create transforms for the data, including normalization w.r.t. mean and standard deviation
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(muhat,sigmahat)])

        test_data = datasets.MNIST(
            root=LOAD_ROOT,
            train=False,
            download=True,
            transform=test_transform
        )

        DATA = 'MNIST'

    else:

        # Create transforms for the data which turns it into grayscale 28x28 images
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(num_output_channels=1),transforms.Resize(size=(28,28))])

        # Load CIFAR-10 data and turn it into data of the same shape as MNIST data
        # for out of distribution testing for equivariance
        test_data = datasets.CIFAR10(
            root=LOAD_ROOT,
            train=False,
            download=True,
            transform=test_transform
        )

        DATA = 'CIFAR10'

    # Create dataloader for test data
    testloader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

    N=members[-1]
    M=member

    # Choose architecture depending on TASK ID
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
        INV = 'SYMINIT'
        FILE_NAME = 'SYMM'
        models = [networks_final.omegaCNN(omega) for i in range(M)]
    else:
        omega[0,0]=1
        '''omega[0,0]=1
        omega[0,1]=1
        omega[1,0]=1
        omega[1,1]=1'''
        FILE_NAME = 'ASYM'
        if syminit:
            INV='SYMINIT'
            models = [networks_final.omegaCNNAsym(omega) for i in range(M)]
        else:
            INV='ASYMINIT'
            models = [networks_final.omegaCNN(omega) for i in range(M)]

    if M==members[-1]:
        indices=range(M)
    else:
        # Load indices
        PATH = os.path.join(LOAD_ROOT3,FILE_NAME+'_'+INV+'_'+str(M)+'_random_indices.npy')
        indices = np.load(PATH)

    # Initialize arrays for saving evaluation data
    loss_array = np.zeros(M)
    acc_array = np.zeros(M)
    osp_mean_array = np.zeros(M)
    osp_std_array = np.zeros(M)
    div_array = np.zeros(M)
    div_max_array = np.zeros(M)

    # Load and evaluate individual models for current epoch
    i=0
    for model in tqdm(models,desc='Members tested:',leave=False):
        model.to(device)
        PATH = os.path.join(LOAD_ROOT2, FILE_NAME+'_'+INV+'_'+str(indices[i])+'_'+str(epoch))
        if device.type == 'cuda':
            model.load_state_dict(torch.load(PATH, weights_only=True))
        else:
            model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device('cpu')))
        model.eval()
        loss, acc, osp_mean, osp_std, div, div_max = test(model,testloader,criterion, device, G)
        loss_array[i] = loss
        acc_array[i] = acc
        osp_mean_array[i] = osp_mean
        osp_std_array[i] = osp_std
        div_array[i] = div
        div_max_array[i] = div_max
        i+=1

    # Save data
    eval_array = np.stack((loss_array,acc_array,osp_mean_array,osp_std_array,div_array,div_max_array))
    PATH = os.path.join(SAVE_ROOT, FILE_NAME+'_'+INV+'_'+str(M)+'_'+'individual_models_for_epoch'+'_'+str(epoch)+'_'+DATA+'_'+'eval_data')
    np.save(PATH,eval_array)

    # Create ensemble model
    ensemble = networks_final.Ensemble(members=models)
    ensemble.to(device)
    ensemble.eval()

    # Evaluate ensemble
    loss, acc, osp_mean, osp_std, div, div_max = test(ensemble,testloader,criterion, device, G)

    # Save data
    ensemble_eval_array = np.array([loss,acc,osp_mean,osp_std,div,div_max])
    PATH = os.path.join(SAVE_ROOT, FILE_NAME+'_'+INV+'_'+str(M)+'_'+'model_ensemble_for_epoch'+'_'+str(epoch)+'_'+DATA+'_'+'eval_data')
    np.save(PATH,ensemble_eval_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that evaluates ensembles"
    )
    parser.add_argument("--sym", required=False, dest = 'symm', action='store_true')
    parser.add_argument("--asym", required=False, dest = 'symm', action='store_false')
    parser.add_argument("--syminit", required=False, dest = 'symminit', action='store_true')
    parser.add_argument("--asyminit", required=False, dest = 'symminit', action='store_false')
    parser.add_argument("--members", required=False, type=int, default=1000)
    parser.add_argument("--epochs", required=False, type=int, default=10)
    parser.add_argument("--allep", required= False, dest ='allepp',action='store_true')
    parser.add_argument("--ep", required= False, dest ='allepp',action='store_false')
    parser.add_argument("--allsize", required= False, dest ='allsizes',action='store_true')
    parser.add_argument("--size", required= False, dest ='allsizes',action='store_false')
    
    args = parser.parse_args()

    sym = args.symm
    syminit = args.symminit
    members = args.members
    epochs = args.epochs
    allae = args.allepp
    allas = args.allsizes
    if allas:
        members = (int(np.ceil(members/100)), int(np.ceil(members/10)), members)
        if members[0]==members[1]:
            members=members[:1]+(members[1]+1,)+members[2:]
        else:
            pass
    else:
        members=(members,)
    if allae:
        epochs = range(epochs+1)
    else:
        epochs = (epochs,)
    for member in members:
        subensembles(members[-1],member,sym,syminit)
        for epoch in epochs:
            for ind in range(2):
                evaluate(sym,syminit,member,epoch,ind)