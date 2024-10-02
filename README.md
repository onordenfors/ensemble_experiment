# ensemble_experiment

This code is supplementary material for the paper "Ensembles provably learn equivariance through data augmentation".

------------------------------------

The directory contains the python files:

ensemble_experiment_final.py
experiment_evaluation_final.py
plotting_final.py
trainer_final.py
equivariance_tester_final.py
networks_final.py
augmentation_final.py
generate_subensembles_final.py

------------------------------------
NOTE: Following these steps and executing the code on your system will create directories
HOW TO PERFORM THE EXPERIMENT:

To run the code, please install the required packages:

pytorch, torchvision, numpy, matplotlib, tqdm

To run the experiment from the paper, follow these steps in order:

Run the file ensemble_experiment_final.py with the flags
    ensemble_experiment_final.py --sym --syminit --members 1000 --epochs 10
    ensemble_experiment_final.py --asym --syminit --members 1000 --epochs 10
    ensemble_experiment_final.py --asym --asyminit --members 1000 --epochs 10

Then run the file experiment_evaluation_final.py with the flags
    experiment_evaluation_final.py --sym --syminit --members 1000 --epochs 10 --allep --size
    experiment_evaluation_final.py --asym --syminit --members 1000 --epochs 10 --allep --size
    experiment_evaluation_final.py --asym --asyminit --members 1000 --epochs 10 --allep --size

Then run the code plotting_final.py with the flags
    plotting_final.py --members 1000 --epochs 10

This will display something similar to Figure 4 from the paper.

Please note that running this code will take a VERY long time on a personal computer or laptop.

Use the flags --members and --epochs to change the number of members/epochs. The initalization can be changed by flags --asymint/--syminit, and the supports with --asym/--sym. Note that --sym will always cause a symmetric initialization

To evaluate a specific epoch, remove the flag --allep when Calling the evaluation script; use the --ep flag to evaluate ensembles for a specific epoch. If one wishes to evaluate performance of subensembles please use the --allsize flag, it will generate and test subensembles of sizes N/10 and N/100 where N is the number of members in total; use the --size flag to only evaluate the entire ensemble and its individual members

To run the experiment for 5x5-filters: Go into the file ensemble_experiment_final.py and comment out lines 74-78, 90, 97 and remove the comments on lines 79-88, 91-94, 98-101, and go into the file experiment_evaluation_final.py and comment out lines 106-110, 122, 131 and remove the comments on lines 111-120, 123-126, 132-135. To run the code for other filter sizes, please modify the corresponding parts of the code.
