import argparse
import numpy as np
import os
import matplotlib.pyplot as plt


def plot(size,epochs):
    PATH = os.getcwd() # Get current directory
    PATH1 = os.path.join(PATH, r'EnsembleExperiment/Evaluation/') # create new directory name
    if not os.path.isdir(PATH1): # if the directory does not already exist
        os.makedirs(PATH1) # make a new directory
    PATH2 = os.path.join(PATH, r'EnsembleExperiment/Figures/') # create new directory name
    if not os.path.isdir(PATH2): # if the directory does not already exist
        os.makedirs(PATH2) # make a new directory

    SAVE_ROOT = './EnsembleExperiment/Figures/'
    LOAD_ROOT2 = './EnsembleExperiment/Evaluation/' # Root for loading results
    LOAD_ROOT3 = './EnsembleExperiment/Evaluation/' # Root for loading results

    DATA = 'MNIST'
    ARCHITECTURE = 'ASYM'
    INV = 'SYMINIT'
    ENSEMBLE_SIZE = size
    EPOCHS = range(epochs+1)

    OSP_ind_skew2=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    OSP_ens_skew2=np.zeros(len(EPOCHS))
    OSP_ens_skew2_std=np.zeros(len(EPOCHS))
    DIV_ens_skew2 = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT3,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT3,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        OSP_ind_skew2[epoch,:] = a[2,:]
        OSP_ens_skew2[epoch] = b[2]
        OSP_ens_skew2_std[epoch] = b[3]
        DIV_ens_skew2[epoch] = b[4]

    acc_skew2 = b[1]

    INV = 'ASYMINIT'
    OSP_ind_skew=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    OSP_ens_skew=np.zeros(len(EPOCHS))
    OSP_ens_skew_std=np.zeros(len(EPOCHS))
    DIV_ens_skew = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        OSP_ind_skew[epoch,:] = a[2,:]
        OSP_ens_skew[epoch] = b[2]
        OSP_ens_skew_std[epoch] = b[3]
        DIV_ens_skew[epoch] = b[4]

    acc_skew = b[1]


    ARCHITECTURE = 'SYMM'
    INV = 'SYMINIT'

    OSP_ind_cross=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    OSP_ens_cross=np.zeros(len(EPOCHS))
    OSP_ens_cross_std=np.zeros(len(EPOCHS))
    DIV_ens_cross = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        OSP_ind_cross[epoch,:] = a[2,:]
        OSP_ens_cross[epoch] = b[2]
        OSP_ens_cross_std[epoch] = b[3]
        DIV_ens_cross[epoch] = b[4]

    acc_cross = b[1]


    DATA = 'CIFAR10'
    ARCHITECTURE = 'ASYM'
    INV = 'SYMINIT'

    cOSP_ind_skew2=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    cOSP_ens_skew2=np.zeros(len(EPOCHS))
    cOSP_ens_skew2_std=np.zeros(len(EPOCHS))
    cDIV_ens_skew2 = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT3,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT3,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        cOSP_ind_skew2[epoch,:] = a[2,:]
        cOSP_ens_skew2[epoch] = b[2]
        cOSP_ens_skew2_std[epoch] = b[3]
        cDIV_ens_skew2[epoch] = b[4]

    INV = 'ASYMINIT'

    cOSP_ind_skew=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    cOSP_ens_skew=np.zeros(len(EPOCHS))
    cOSP_ens_skew_std=np.zeros(len(EPOCHS))
    cDIV_ens_skew = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        cOSP_ind_skew[epoch,:] = a[2,:]
        cOSP_ens_skew[epoch] = b[2]
        cOSP_ens_skew_std[epoch] = b[3]
        cDIV_ens_skew[epoch] = b[4]

    ARCHITECTURE = 'SYMM'
    INV = 'SYMINIT'

    cOSP_ind_cross=np.zeros((len(EPOCHS),ENSEMBLE_SIZE))
    cOSP_ens_cross=np.zeros(len(EPOCHS))
    cOSP_ens_cross_std=np.zeros(len(EPOCHS))
    cDIV_ens_cross = np.zeros(len(EPOCHS))

    for epoch in EPOCHS:
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_individual_models_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        a = np.load(PATH)
        PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data.npy')
        b = np.load(PATH)
        cOSP_ind_cross[epoch,:] = a[2,:]
        cOSP_ens_cross[epoch] = b[2]
        cOSP_ens_cross_std[epoch] = b[3]
        cDIV_ens_cross[epoch] = b[4]

    fig, ax = plt.subplots(2, 2, sharex=True)


    ax[0,0].fill_between(EPOCHS,np.quantile(OSP_ind_skew,0.975,axis=1), np.quantile(OSP_ind_skew,0.025,axis=1),linestyle='dashdot',color='#2ca02c',alpha = 0.3)
    ax[0,0].fill_between(EPOCHS,np.quantile(OSP_ind_skew2,0.975,axis=1), np.quantile(OSP_ind_skew2,0.025,axis=1),linestyle='dashed',color='#ff7f0e',alpha = 0.3)


    ax[0,0].plot(EPOCHS,OSP_ens_cross,color='#1f77b4',linestyle='solid', label=r'$\mathcal{L}^{\mathrm{sym}}$')
    ax[0,0].plot(EPOCHS,OSP_ens_skew2,color='#ff7f0e',linestyle='dashed', label=r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[0,0].plot(EPOCHS,OSP_ens_skew,color='#2ca02c',linestyle='dashdot', label=r'$\mathcal{L}^{\mathrm{as}}$')
    ax[0,0].fill_between(EPOCHS,np.quantile(OSP_ind_cross,0.975,axis=1), np.quantile(OSP_ind_cross,0.025,axis=1),color='#1f77b4',linestyle='solid',alpha = 0.3)

    ax[1,0].plot(EPOCHS,np.log10(DIV_ens_cross),linestyle='solid',color='#1f77b4', label = r'$\mathcal{L}^{\mathrm{sym}}$')
    ax[1,0].plot(EPOCHS,np.log10(DIV_ens_skew2),linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[1,0].plot(EPOCHS,np.log10(DIV_ens_skew),linestyle='dashdot',color='#2ca02c', label = r'$\mathcal{L}^{\mathrm{as}}$')
    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel(r'$\log_{10}$-Divergence')
    ax[0,0].set_ylabel('OSP')
    ax[0,0].set_title('MNIST test data')
    ax[0,0].legend()
    ax[1,0].legend()


    ax[0,1].fill_between(EPOCHS,np.quantile(cOSP_ind_skew,0.975,axis=1), np.quantile(cOSP_ind_skew,0.025,axis=1),linestyle='dashdot',color='#2ca02c',alpha = 0.3)
    ax[0,1].fill_between(EPOCHS,np.quantile(cOSP_ind_skew2,0.975,axis=1), np.quantile(cOSP_ind_skew2,0.025,axis=1),linestyle='dashed',color='#ff7f0e',alpha = 0.3)


    ax[0,1].plot(EPOCHS,cOSP_ens_cross,color='#1f77b4',linestyle='solid', label=r'$\mathcal{L}^{\mathrm{sym}}$')
    ax[0,1].plot(EPOCHS,cOSP_ens_skew2,color='#ff7f0e',linestyle='dashed', label=r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[0,1].plot(EPOCHS,cOSP_ens_skew,color='#2ca02c',linestyle='dashdot', label=r'$\mathcal{L}^{\mathrm{as}}$')
    ax[0,1].fill_between(EPOCHS,np.quantile(cOSP_ind_cross,0.975,axis=1), np.quantile(cOSP_ind_cross,0.25,axis=1),color='#1f77b4',alpha = 0.3)

    ax[1,1].plot(EPOCHS,np.log10(cDIV_ens_cross),linestyle='solid',color='#1f77b4', label = r'$\mathcal{L}^{\mathrm{sym}}$')
    ax[1,1].plot(EPOCHS,np.log10(cDIV_ens_skew2),linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[1,1].plot(EPOCHS,np.log10(cDIV_ens_skew),linestyle='dashdot',color='#2ca02c', label = r'$\mathcal{L}^{\mathrm{as}}$')
    ax[1,1].set_xlabel('Epoch')
    ax[0,1].set_title('CIFAR—10 test data')
    ax[0,1].legend()
    ax[1,1].legend()

    #fig.suptitle('Metrics for Ensembles with 1000 Members')

    ax[0,0].set_yticks(np.arange(1,4.5,0.5))
    ax[0,1].set_yticks(np.arange(1,4.5,0.5))
    ax[1,0].set_yticks(np.arange(-3,2,1))
    ax[1,1].set_yticks(np.arange(-3,2,1))
    plt.xticks((0,1,2,3,4,5,6,7,8,9,10))
    plt.rcParams['figure.dpi']=600
    plt.savefig(os.path.join(SAVE_ROOT, 'cross_vs_skew_v2_1.svg'), format='svg')
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that trains ensemble members"
    )

    parser.add_argument("--members", required=False, type=int, default=1000)
    parser.add_argument("--epochs", required=False, type=int, default=10)
    
    args = parser.parse_args()

    members = args.members
    epochs = args.epochs
    plot(members,epochs)