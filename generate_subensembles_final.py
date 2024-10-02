import numpy as np
import os

def subensembles(N,M,sym, syminit):
    PATH = os.getcwd() # Get current directory
    PATH1 = os.path.join(PATH, r'EnsembleExperiment/Subensembles/') # create new directory name
    if not os.path.isdir(PATH1): # if the directory does not already exist
        os.makedirs(PATH1) # make a new directory

    SAVE_ROOT = './EnsembleExperiment/Subensembles/' # Root for saving data
    if sym:
        INV='SYMINIT'
        FILE_NAME = 'SYMM'
    else:
        if syminit:
            INV='SYMINIT'
            FILE_NAME = 'ASYM'
        else:
            INV ='ASYMINIT'
            FILE_NAME = 'ASYM'

    indices = np.random.choice(range(N), M, replace=False)
    np.save(os.path.join(SAVE_ROOT,FILE_NAME+'_'+INV+'_'+str(M)+'_random_indices'),indices)