import argparse
import numpy as np
import os
import matplotlib.pyplot as plt


def plot(members,epoch,group,bootstraps):
    PATH = os.getcwd() # Get current directory
    PATH1 = os.path.join(PATH,'EnsembleExperiment',group,'Evaluation') # create new directory name
    if not os.path.isdir(PATH1): # if the directory does not already exist
        os.makedirs(PATH1) # make a new directory
    PATH2 = os.path.join(PATH,'EnsembleExperiment',group,'Figures') # create new directory name
    if not os.path.isdir(PATH2): # if the directory does not already exist
        os.makedirs(PATH2) # make a new directory

    SAVE_ROOT = os.path.join('EnsembleExperiment',group,'Figures')
    LOAD_ROOT2 = os.path.join('EnsembleExperiment',group,'Evaluation') # Root for loading results
    LOAD_ROOT3 = LOAD_ROOT2 # Root for loading results

    DATA = 'MNIST'
    ARCHITECTURE = 'ASYM'
    INV = 'SYMINIT'
    

    OSP_ens_skew2 =np.zeros((len(members),bootstraps))
    DIV_ens_skew2 = np.zeros((len(members),bootstraps))
    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT3,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            OSP_ens_skew2[j,bootstrap] = b[2]
            DIV_ens_skew2[j,bootstrap] = b[4]

    INV = 'ASYMINIT'
    OSP_ens_skew=np.zeros((len(members),bootstraps))
    DIV_ens_skew = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            OSP_ens_skew[j,bootstrap] = b[2]
            DIV_ens_skew[j,bootstrap] = b[4]

    ARCHITECTURE = 'SYMM'
    INV = 'SYMINIT'
    OSP_ens_cross=np.zeros((len(members),bootstraps))
    DIV_ens_cross = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            OSP_ens_cross[j,bootstrap] = b[2]
            DIV_ens_cross[j,bootstrap] = b[4]

    ARCHITECTURE = 'CNN'
    INV = ''
    OSP_ens_cnn = np.zeros((len(members),bootstraps))
    DIV_ens_cnn = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            OSP_ens_cnn[j,bootstrap] = b[2]
            DIV_ens_cnn[j,bootstrap] = b[4]


    DATA = 'CIFAR10'
    ARCHITECTURE = 'ASYM'
    INV = 'SYMINIT'

    cOSP_ens_skew2 = np.zeros((len(members),bootstraps))
    cDIV_ens_skew2 = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            cOSP_ens_skew2[j,bootstrap] = b[2]
            cDIV_ens_skew2[j,bootstrap] = b[4]

    INV = 'ASYMINIT'
    cOSP_ens_skew = np.zeros((len(members),bootstraps))
    cDIV_ens_skew = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            cOSP_ens_skew[j,bootstrap] = b[2]
            cDIV_ens_skew[j,bootstrap] = b[4]

    ARCHITECTURE = 'SYMM'
    INV = 'SYMINIT'

    cOSP_ens_cross=np.zeros((len(members),bootstraps))
    cDIV_ens_cross = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            cOSP_ens_cross[j,bootstrap] = b[2]
            cDIV_ens_cross[j,bootstrap] = b[4]
    
    ARCHITECTURE = 'CNN'
    INV = ''
    cOSP_ens_cnn = np.zeros((len(members),bootstraps))
    cDIV_ens_cnn = np.zeros((len(members),bootstraps))

    for j,ENSEMBLE_SIZE in enumerate(members):
        for bootstrap in range(bootstraps):
            PATH = os.path.join(LOAD_ROOT2,ARCHITECTURE+'_'+INV+'_'+str(ENSEMBLE_SIZE)+'_model_ensemble_for_epoch_'+str(epoch)+'_'+DATA+'_eval_data'+str(bootstrap)+'.npy')
            b = np.load(PATH)
            cOSP_ens_cnn[j,bootstrap] = b[2]
            cDIV_ens_cnn[j,bootstrap] = b[4]

    fig, ax = plt.subplots(2, 2, sharex=True)

    
    ax[0,0].errorbar(x= members,y = OSP_ens_cross[:,:].mean(1),yerr = OSP_ens_cross[:,:].std(1), label = r'$\mathcal{L}^{\mathrm{sym}}$', linestyle='solid', color = '#1f77b4')
    ax[0,0].errorbar(x= members,y = OSP_ens_skew[:,:].mean(1),yerr = OSP_ens_skew[:,:].std(1), label=r'$\mathcal{L}^{\mathrm{as}}$', linestyle='dashdot',color = '#2ca02c' )
    ax[0,0].errorbar(x= members,y = OSP_ens_skew2[:,:].mean(1),yerr = OSP_ens_skew2[:,:].std(1), linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[0,0].errorbar(x= members,y = OSP_ens_cnn[:,:].mean(1),yerr = OSP_ens_cnn[:,:].std(1), linestyle='dashed',color='#b53737', label = r'$\mathcal{L}^{\mathrm{cnn}}$')
    ax[0,0].legend()
    ax[0,0].set_xscale('log')
    ax[0,0].set_ylabel('OSP')
    ax[0,0].set_title('MNIST test data')

    ax[0,1].errorbar(x= members,y = cOSP_ens_cross[:,:].mean(1),yerr = cOSP_ens_cross[:,:].std(1), label = r'$\mathcal{L}^{\mathrm{sym}}$', linestyle='solid', color = '#1f77b4')
    ax[0,1].errorbar(x= members,y = cOSP_ens_skew[:,:].mean(1),yerr = cOSP_ens_skew[:,:].std(1), label=r'$\mathcal{L}^{\mathrm{as}}$', linestyle='dashdot',color = '#2ca02c' )
    ax[0,1].errorbar(x= members,y = cOSP_ens_skew2[:,:].mean(1),yerr = cOSP_ens_skew2[:,:].std(1), linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[0,1].errorbar(x= members,y = cOSP_ens_cnn[:,:].mean(1),yerr = cOSP_ens_cnn[:,:].std(1), linestyle='dashed',color='#b53737', label = r'$\mathcal{L}^{\mathrm{cnn}}$')
    ax[0,1].legend()
    ax[0,1].set_xscale('log')
    ax[0,1].set_title('CIFAR test data')


    ax[1,0].errorbar(x= members,y = DIV_ens_cross[:,:].mean(1),yerr = DIV_ens_cross[:,:].std(1), label = r'$\mathcal{L}^{\mathrm{sym}}$', linestyle='solid', color = '#1f77b4')
    ax[1,0].errorbar(x= members,y = DIV_ens_skew[:,:].mean(1),yerr = DIV_ens_skew[:,:].std(1), label=r'$\mathcal{L}^{\mathrm{as}}$', linestyle='dashdot',color = '#2ca02c' )
    ax[1,0].errorbar(x= members,y = DIV_ens_skew2[:,:].mean(1),yerr = DIV_ens_skew2[:,:].std(1), linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[1,0].errorbar(x= members,y = DIV_ens_cnn[:,:].mean(1),yerr = DIV_ens_cnn[:,:].std(1), linestyle='dashed',color='#b53737', label = r'$\mathcal{L}^{\mathrm{cnn}}$')
    ax[1,0].legend()
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_ylabel('DIV')
    ax[1,0].set_xlabel('Ensemble members')
    
    ax[1,1].errorbar(x= members,y = cDIV_ens_cross[:,:].mean(1),yerr = cDIV_ens_cross[:,:].std(1), label = r'$\mathcal{L}^{\mathrm{sym}}$', linestyle='solid', color = '#1f77b4')
    ax[1,1].errorbar(x= members,y = cDIV_ens_skew[:,:].mean(1),yerr = cDIV_ens_skew[:,:].std(1), label=r'$\mathcal{L}^{\mathrm{as}}$', linestyle='dashdot',color = '#2ca02c' )
    ax[1,1].errorbar(x= members,y = cDIV_ens_skew2[:,:].mean(1),yerr = cDIV_ens_skew2[:,:].std(1), linestyle='dashed',color='#ff7f0e', label = r'$\mathcal{L}^{\mathrm{as}}$, $\rho(g)A$~$A$')
    ax[1,1].errorbar(x= members,y = cDIV_ens_cnn[:,:].mean(1),yerr = cDIV_ens_cnn[:,:].std(1), linestyle='dashed',color='#b53737', label = r'$\mathcal{L}^{\mathrm{cnn}}$')
    ax[1,1].legend()
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_title('CIFAR test data')
    ax[1,1].set_xlabel('Ensemble members')
    
    
    """
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
    """
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that trains ensemble members"
    )

    parser.add_argument("--members", required=False, type=int, default=1000)
    parser.add_argument("--epochs", required=False, type=int, default=10)
    parser.add_argument("--bootstraps",required = False, type = int, default = 1)
    parser.add_argument("--C4", required=False, dest = 'group4', action='store_true')
    parser.add_argument("--C16", required=False, dest = 'group4', action='store_false')
    
    args = parser.parse_args()

    if args.group4:
        group = 'C4'
    else:
        group = 'C16'

    members = args.members
    epochs = args.epochs
    bootstraps = args.bootstraps

    pot_members = [5,10,25,50,75,100,250,500,1000]
    memberss = []
    k=0
    while k<len(pot_members) and pot_members[k]<=members:
        memberss+=[pot_members[k]]
        k=k+1
    members = memberss

    plot(members,epochs,group,bootstraps)