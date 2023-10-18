## Introduction
This repository containts the code capable of reproducing the results of the paper **How a student becomes a teacher:
learning and forgetting through Spectral methods** by *L.Giambagli, L.Buffoni, L.Chicchi, D.Fanelli*.

To install all dependencies run `pip install -r requirements.txt`

The so called Spectral Layer and the Structural Pruning functions employed can be installed using `pip install spectraltools`. 
For further details go to [SpectralTools](https://github.com/Jamba15/SpectralTools) GitHub page.

### Plot the results
To plot the result in the paper run the python file `main_results.py`. The code contains three different section that 
can be commented depending on the desired results. More specifically, the *first section* reproduces the histograms of the
scalars $\mathcal{M}$ and $\mathcal{L}$ for various student hidden layer sizes.
The *second section* reproduces the plots of the average dimension of the student hidden layer aggregating the 
30 trials. The *third section* reproduces the plots of the average test accuracy of the student network aggregating the
runs, again, over the 30 trials.

### Run new experiments
The file `main_run_experiments.py` contains the code to run the experiments for the different student hidden layer sizes basing
on the configuration file `Configurations/config.json`. The results are always saved in the folder `Results` with subdirectory structure 
given in the configuration.

### 
In order to run the code properly several packages might be needed. The file `requirements.txt` contains the list of the
packages used in the development of the code. The code has been tested with Python 3.10.0.
