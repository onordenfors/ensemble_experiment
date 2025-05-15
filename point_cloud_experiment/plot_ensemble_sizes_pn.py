import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

PATH = os.getcwd() # Get current directory
PATH1 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_results/') # create new directory name
PATH2 = os.path.join(PATH, r'ensemble_pointcloud_experiment/saved_results_ood/') # create new directory name
if not os.path.isdir(PATH1): # if the directory does not already exist
    os.makedirs(PATH1) # make a new directory
else:
    pass
if not os.path.isdir(PATH2): # if the directory does not already exist
    os.mkdir(PATH2) # make a new directory
else:
    pass
LOAD_ROOT = './ensemble_pointcloud_experiment/saved_results/' # Root for loading dataset
LOAD_ROOT2 = './ensemble_pointcloud_experiment/saved_results_ood/' # Root for saving data

ensembles_sizes = [5, 10, 25, 50, 100, 250, 500, 1000]

results_osp = np.zeros((len(ensembles_sizes),30))
results_div = np.zeros((len(ensembles_sizes),30))
results_osp_ood = np.zeros((len(ensembles_sizes),30))
results_div_ood = np.zeros((len(ensembles_sizes),30))

for i,size in enumerate(ensembles_sizes):
    for ensemble in range(30):
        run_osp = np.load(os.path.join(LOAD_ROOT,'metric_osp_'+str(ensemble)+'_ensemble_size_'+str(size))+'.npy')
        run_div = np.load(os.path.join(LOAD_ROOT,'metric_div_'+str(ensemble)+'_ensemble_size_'+str(size))+'.npy')
        results_osp[i,ensemble] = run_osp.sum()/908*100
        results_div[i,ensemble] = run_div

        run_osp = np.load(os.path.join(LOAD_ROOT2,'metric_osp_'+str(ensemble)+'_ensemble_size_'+str(size))+'.npy')
        run_div = np.load(os.path.join(LOAD_ROOT2,'metric_div_'+str(ensemble)+'_ensemble_size_'+str(size))+'.npy')
        results_osp_ood[i,ensemble] = run_osp.sum()/len(run_osp)*100
        results_div_ood[i,ensemble] = run_div

fig,ax = plt.subplots(2,2)

ax[0,0].errorbar(ensembles_sizes,y=results_osp.mean(1),yerr=results_osp.std(1),color = '#1f77b4')
ax[0,0].set_xscale('log')
ax[1,0].errorbar(ensembles_sizes,y=results_div.mean(1),yerr=results_div.std(1),color = '#1f77b4')
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')

ax[0,1].errorbar(ensembles_sizes,y=results_osp_ood.mean(1),yerr=results_osp_ood.std(1),color = '#1f77b4')
ax[0,1].set_xscale('log')
ax[1,1].errorbar(ensembles_sizes,y=results_div_ood.mean(1),yerr=results_div_ood.std(1),color = '#1f77b4')
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')

plt.rcParams['text.usetex'] = False
plt.rcParams.update({'font.size':15})

ax[0,0].set_ylabel('OSP',fontsize=15)
ax[0,0].set_title('In distr. data')

ax[0,1].set_title('Out of distr. data')

ax[1,0].set_ylabel('KL-DIV',fontsize=15)

ax[1,0].set_xlabel('Ensemble members',fontsize=15)
ax[1,1].set_xlabel('Ensemble members',fontsize=15)

ax[0,0].tick_params(labelsize=10)
ax[0,1].tick_params(labelsize=10)
ax[1,0].tick_params(labelsize=10)
ax[1,1].tick_params(labelsize=10)

p = .001/(7*4)


table_string = "\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|} \n"
table_string += "\\hline  Dataset & Metric & 5 & 10 & 25 & 50 & 100 & 250 & 500 & 1000 \\\ \n"
table_string += " \\hline \\multirow{2}{*}{In dist.} & OSP "

for i,size in enumerate(ensembles_sizes):
    table_string += ' & ' + '%.2f'%results_osp[i,:].mean()

table_string += " \\\ \n"
table_string += "\\cline{2-10} &  $\\log(\\text{KL-DIV})$"

for i,size in enumerate(ensembles_sizes):
    table_string +=  ' & ' + '%.2f'%np.log10(results_div[i,:].mean()) 

table_string += "\\\ \n"
table_string += " \\hline \\multirow{2}{*}{Out of dist.} & OSP "

for i,size in enumerate(ensembles_sizes):
    table_string += ' & ' + '%.2f' %results_osp_ood[i,:].mean()

table_string += " \\\ \n"
table_string += "\\cline{2-10} & $\\log(\\text{KL-DIV})$"

for i,size in enumerate(ensembles_sizes):
    table_string +=  ' & ' + '%.2f'%np.log10(results_div_ood[i,:].mean()) 

table_string += "\n \\hline \\end{tabular}"

print(table_string)



for i,size in enumerate(ensembles_sizes):   
    tres= ttest_ind(results_osp[-1,:],results_osp[i,:],equal_var=False,alternative='greater')
    if tres.pvalue<p:
        print('OSP of ' +  str(size) +' is significantly smaller than 1000')

    tres= ttest_ind(results_div[-1,:],results_div[i,:],equal_var=False,alternative='less')
    if tres.pvalue<p:
        print('div of ' + str(size) + ' is significantly larger than 1000')

for i,size in enumerate(ensembles_sizes):   
    tres= ttest_ind(results_osp_ood[-1,:],results_osp_ood[i,:],equal_var=False,alternative='greater')
    if tres.pvalue<p:
        print('OSP of ' +  str(size) +' is significantly smaller than 1000 (ood)')

    tres= ttest_ind(results_div_ood[-1,:],results_div_ood[i,:],equal_var=False,alternative='less')
    if tres.pvalue<p:
        print('div of ' + str(size) + ' is significantly larger than 1000 (ood)')




plt.suptitle('ModelNet',fontsize=20)

plt.show()
