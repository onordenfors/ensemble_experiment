import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import os
from random import choices
from tqdm import tqdm
import scipy as sp

def metrics(member_list, load_path):

    G=100
    osp = 0
    div = 0

    outputs = [np.load(os.path.join(load_path, 'logits'+'_'+'model'+'_'+str(member)+'.npy')) for member in member_list]
    mean_output = sum(outputs)/len(outputs)
    probs = sp.special.softmax(mean_output, axis=2)
    log_probs = sp.special.log_softmax(mean_output, axis=2)


    for g in range(1,G+1):
        osp += (mean_output[:,0,:].argmax(-1)==mean_output[:,g,:].argmax(-1))
        div += F.kl_div(torch.from_numpy(log_probs[:,g,:]), torch.from_numpy(probs[:,0,:]),reduction='batchmean').item() + F.kl_div(torch.from_numpy(log_probs[:,0,:]), torch.from_numpy(probs[:,g,:]),reduction='batchmean').item()
    osp = osp/100
    div = div/100
    return osp, div

def subensemble(n):
    member_list = choices(range(1000),k=n)
    return member_list

PATH = os.getcwd() # Get current directory
PATH1 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_results/') # create new directory name
PATH2 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_models/') # create new directory name
PATH3 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_logits/') # create new directory name
PATH4 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_logits_ood/') # create new directory name
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

LOAD_ROOT = './ensemble_pointcloud_experiment/saved_logits/'
LOAD_ROOT2 = './ensemble_pointcloud_experiment/saved_logits_ood/'
SAVE_PATH =  './ensemble_pointcloud_experiment/saved_results/'
SAVE_PATH2 = './ensemble_pointcloud_experiment/saved_results/'

ensembles_sizes = [5, 10, 25, 50, 100, 250, 500, 1000]

for size in tqdm(ensembles_sizes):
    for ensemble in tqdm(range(30)):
        member_list = subensemble(size)
        osp, div = metrics(member_list, LOAD_ROOT)
        osp2, div2 = metrics(member_list, LOAD_ROOT2)
        np.save(os.path.join(SAVE_PATH,'metric_osp_'+str(ensemble)+'_ensemble_size_'+str(size)),osp)
        np.save(os.path.join(SAVE_PATH,'metric_div_'+str(ensemble)+'_ensemble_size_'+str(size)),div)
        np.save(os.path.join(SAVE_PATH2,'metric_osp_'+str(ensemble)+'_ensemble_size_'+str(size)),osp2)
        np.save(os.path.join(SAVE_PATH2,'metric_div_'+str(ensemble)+'_ensemble_size_'+str(size)),div2)