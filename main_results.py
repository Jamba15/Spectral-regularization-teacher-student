from proj_functions import LoadConfig, ResultsHandler
from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt
from os.path import join, exists

# Check if folder Results exists, if not download the results from the google drive
if not exists('./Results'):
    print('Results folder not found, downloading from google drive')
    # Download the results shown in the paper from the google drive and create the Results folder
    gdd.download_file_from_google_drive(file_id='1twujBUYvS7lhzu_LrOlUeDQ6qk24ABw6',
                                        dest_path='./Results/L2_all.zip',
                                        unzip=True)
else:
    print('Pre-existing Results folder found')

# Load the results
results = ResultsHandler(config=LoadConfig('multirunL2_all.yml'))

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
# results.LossComparePlot(save=False)
