import ray
from proj_functions import SpectralDirectSameInit, OldResultsRemover, TrialTerminationReporter
from ray import tune


"""
This is the main file to run the experiments. It is divided in two parts: the first one is to run the experiments, the
second one is to plot the results. The just_plot variable is used to switch between the two parts and is True by default.
The trial_info dictionary contains the information about the configuration file to use and the hyperparameters (if a swipe along different 
values from the one indicated in the configuration file is needed). The delete_old_results variable is used to delete the
old results in the folder indicated by the configuration_name.
The trials are run using the ray library, which is a distributed computing library. The number of samples is the number
of trials to run per hyperparameter configuration.
"""

trial_info = {'configuration_name': 'multirunL2_all.yml',
              'hyperparameters': {'units': tune.grid_search(
                  [20, 60, 200])}}  # , 800, 1000, 1500])  # [10, 20, 200, 500, 700, 1000])}}

# If true, it will delete the old results in the folder indicated by the configuration_name
delete_old_results = False

ray.init(log_to_driver=False)
if delete_old_results:
    OldResultsRemover(trial_info['configuration_name'])

num_samples = 10
reporter = TrialTerminationReporter()

tune.run(tune.with_parameters(SpectralDirectSameInit, save_results=True),
         num_samples=num_samples,
         resources_per_trial={'cpu': 0.3, 'gpu': 1 / 10},
         progress_reporter=reporter,
         config=trial_info)
