import ray
from proj_functions import SpectralDirectSameInit, LoadConfig, ResultsHandler, OldResultsRemover
from MyLibrary import TrialTerminationReporter
from ray import tune

import matplotlib.pyplot as plt
from os.path import join

"""
This is the main file to run the experiments. It is divided in two parts: the first one is to run the experiments, the
second one is to plot the results. The just_plot variable is used to switch between the two parts and is True by default.
The trial_info dictionary contains the information about the configuration file to use and the hyperparameters (if a swipe along different 
values from the one indicated in the configuration file is needed). The delete_old_results variable is used to delete the
old results in the folder indicated by the configuration_name.
The trials are run using the ray library, which is a distributed computing library. The number of samples is the number
of trials to run per hyperparameter configuration.
"""

# If true, it will only plot the results
just_plot = True

trial_info = {'configuration_name': 'multirunL2_all.yml',
              'hyperparameters': {'units': tune.grid_search(
                  [20, 60, 200])}}  # , 800, 1000, 1500])  # [10, 20, 200, 500, 700, 1000])}}

# If true, it will delete the old results in the folder indicated by the configuration_name
delete_old_results = False


if not just_plot:
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
else:
    results = ResultsHandler(config=LoadConfig(trial_info['configuration_name']))
    """
    This is the code to plot the eigenvalues histogram"""
    to_plot = [10, 40, 500]
    fig, axs = plt.subplots(1, 3, figsize=(7, 3), sharex=True, sharey=True, dpi=300)
    axs = axs.flatten()
    for i, units in enumerate(to_plot):
        results.EigenvaluesHistogram(conditions_dictionary={'units': units},
                                     ax=axs[i],
                                     cutoff=0.1,
                                     title='units={}'.format(units),
                                     plot_nonzero=True)
    plt.tight_layout()
    plt.savefig(join(results.results_path, 'eigenvalues_histogram.png'))
    plt.show()
    """
    This is the code to plot the layerwise average dimension"""
    # results.AverageDimension(save=False)

    """
    This is the code to plot the loss"""
    # Plot the loss
    # results.LossComparePlot(save=False)
