import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from proj_functions import LoadConfig, ResultsHandler, NNTeacher, DatasetMaker
from proj_functions import MSERamp, FlattenList


class NetworkAnalysis:
    """
    This class is used to perform the analysis of the network. It is used to plot the eigenvalues histogram, the
    layerwise average dimension and the path analysis.
    """
    def __init__(self, config):
        self.config = config
        self.results = ResultsHandler(config)

    def PathAnalysis(self, units, layer_type,
                     full=False,
                     with_teacher=True,
                     ax=None):

        out = self.results.ExtractParameters(condition_dictionary=dict(units=units,
                                                                       layer_type=layer_type))

        lay_1 = []
        lay_2 = []
        lay_1_full = []
        lay_2_full = []

        for i in range(len(out)):
            eigs = out[i].get('eigenvalues')
            filter = (eigs / eigs.max()) > 0.05
            diag_end = out[i].get('diag_end')

            direct = -out[i].get('base') * diag_end
            second_layer = out[i].get('second_layer')
            lay_1_full.append(direct)
            lay_2_full.append(second_layer)
            lay_1.append(direct[:, filter[0, :]])
            lay_2.append(second_layer[filter[0, :], :])

        teacher = NNTeacher(self.config['Dataset'])
        t1 = teacher.layers[1].base.numpy()
        t2 = teacher.layers[2].base.numpy()

        T = []
        for first_i, first_j in product(range(t1.shape[0]), range(t1.shape[1])):
            for second_j in range(t2.shape[1]):
                T.append(t1[first_i, first_j] * t2[first_j, second_j])
        T = np.sort(np.array(T))

        S_all = []
        for model_index in range(len(lay_1)):
            if full:
                s1 = lay_1_full[model_index]
                s2 = lay_2_full[model_index]
            else:
                s1 = lay_1[model_index]
                s2 = lay_2[model_index]

            S = []
            for first_i, first_j in product(range(s1.shape[0]), range(s1.shape[1])):
                for second_j in range(s2.shape[1]):
                    S.append(s1[first_i, first_j] * s2[first_j, second_j])

            S = np.sort(np.array(S))
            S_all.append(S)

        if layer_type == 'Spectral':
            stud_label = 'Student ' if full else 'Student (Pruned) '
        else:
            stud_label = 'Student '

        # Plotting S and T. S is plotted with the average and une standard deviation as a shaded region.
        # considering that the dimensions of every S are different, we need to average them in a smart way
        # to avoid errors: we will map them into a function defined on the same domain (0,1) using the largest
        # as a reference. Then we will interpolate the others and average them.
        # The same will be done for the standard deviation.
        # The interpolation is done using the scipy.interpolate.interp1d function
        # First we need to find the largest S
        max_len = max([len(S) for S in S_all])
        # Then we need to map all the S into the same domain
        S_mapped = []
        for S in S_all:
            S_mapped.append(np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(S)), S))

        # Now we can average them
        S_mean = np.mean(S_mapped, axis=0)
        S_std = np.std(S_mapped, axis=0)

        # Plotting
        ax.fill_between(np.linspace(0, 1, len(S_mean)), S_mean - S_std, S_mean + S_std,
                        alpha=0.2,
                        label=stud_label + layer_type)

        ax.plot(np.linspace(0, 1, len(S_mean)), S_mean,
                linewidth=2,
                alpha=0.8,
                color='blue',
                label=stud_label + layer_type)

        if with_teacher:
            ax.plot(np.linspace(0, 1, len(T)), T, color='orange', label='Teacher')

        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title('Path Analysis' + r' ($n_\lambda \simeq 20$)'.format(units),
                     fontsize=18,
                     fontdict={'fontname': 'serif'})

        ax.set_xlabel('A.U.',
                      fontsize=16,
                      fontdict={'fontname': 'serif'})
        ax.set_ylabel('Path Magnitude', fontsize=16, fontdict={'fontname': 'serif'})
        ax.tick_params(axis='both', which='major', labelsize=14)
        return ax

    def MSEAnalysis(self, units_list, save=False):
        conf = self.config

        model_list = []
        for i in units_list:
            model_list.append(self.results.ExtractModels(condition_dictionary=dict(units=i,
                                                                                   layer_type='Spectral'))[0])

        all_obs = []
        all_order = []
        for k in range(len(model_list)):
            obs, order = MSERamp(model_list[k], conf)
            all_obs.append(obs)
            all_order.append(order)

        title = 'MSE'
        plt.figure(figsize=(6, 4))
        for k in range(len(model_list)):
            plt.plot(all_order[k], all_obs[k],
                     '-o',
                     label='{}'.format(model_list[k].layers[1].units),
                     linewidth=2,
                     alpha=0.7)
        # Insert labels and title
        plt.title(title, fontsize=16, fontdict={'fontname': 'serif'})
        # plt.ylim(0, 15)
        plt.xlabel(r'$n_\lambda-h_T$', fontsize=16, fontdict={'fontname': 'serif'})
        plt.ylabel(r'$\Delta_{MSE}$', fontsize=16, fontdict={'fontname': 'serif'})

        # fit the best exponential law in the interval < 0 using scipy
        xdata = FlattenList([all_order[k] for k in range(len(model_list))])
        xdata = np.array(xdata)
        # print(xdata)
        posx = xdata < 1
        xdata = xdata[posx]
        ydata = FlattenList([all_obs[k] for k in range(len(model_list))])
        ydata = np.array(ydata)
        ydata = ydata[posx]

        plt.yscale('log')
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize=10)
        if save:
            plt.savefig('Phase transition critical point 0.png'.format(title), dpi=300)
        plt.show()
        plt.close()
        return all_obs, all_order, xdata, ydata


if __name__ == '__main__':
    """
    This script is used to run the analysis of the network. It is possible to run the analysis on the teacher,
    on the student, or on both. The analysis can be run on the full network or on the pruned network. The configuration 
    file used is the one with 'analysis' in the name. This is due to the fact that the number of parameters saved are more
    and, therefore, the process is slower.
    """
    config = LoadConfig('multirunL2_all_analysis.yml')
    teacher = NNTeacher(config['Dataset'])
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    index = 0
    units = 200

    ax = NetworkAnalysis(config).PathAnalysis(units=units,
                                              layer_type='Spectral',
                                              # model_index=0,
                                              full=False,
                                              with_teacher=True,
                                              ax=ax)

    # datas = []
    # for configuration in ['multirunL2_hard_analysis.yml']: #,'multirunL2_hard100_analysis.yml']:
    #     config = LoadConfig(configuration)
    #     results = ResultsHandler(config)
    #     all_obs, all_order, x, y = NetworkAnalysis(config).MSEAnalysis(units_list=[ 200],save=True)
    #     datas.append(dict(x=x, y=y, teacher=config['Dataset']['teacher_hidden']))
    plt.tight_layout()
    plt.savefig('Path Analysis Spectral.png', dpi=300)
    plt.show()

# %% Fit exponential
# from scipy.optimize import curve_fit
#
#
# def func(x, a, b, c):
#     """Exponential function for fitting."""
#     return a * np.exp(-b * x) + c
#
#
# plt.figure(figsize=(6, 4), dpi=200)
# plt.title(r'Normalized $\Delta_{MSE}$', fontsize=18, fontdict={'fontname': 'serif'})
#
# for data_dict in datas:
#     xdata = data_dict['x']
#     ydata = data_dict['y']
#     xdata = xdata / data_dict['teacher']
#     ydata = ydata / ydata.max()
#
#     plt.plot(xdata, ydata, 'o', label='Teacher {}'.format(data_dict['teacher']))
#     # get color of the last plot
#     color = plt.gca().lines[-1].get_color()
#     popt, pcov = curve_fit(func, xdata, ydata)
#     xs = np.linspace(-1, -0, 100)
#     # plot with dashed line and black color
#     lbl = '{:.1E} * exp(-{:.1E} * x) + {:.1E}'.format(*popt)
#     plt.plot(xs, func(xs, *popt), '--', alpha=0.7, linewidth=2, color=color, label=lbl)
# plt.ylim(0, 1.1)
# plt.ylabel(r'$\Delta_{MSE}/\Delta_{MSE}^{max}$', fontsize=16, fontdict={'fontname': 'serif'})
# plt.xlabel(r'$n_\lambda/h_T-1$', fontsize=16, fontdict={'fontname': 'serif'})
# plt.legend()
# plt.tight_layout()
# plt.savefig('MSE critical point exponential fit.png', dpi=200)
# plt.show()
# print function with 2 digits
# plt.legend(fontsize=13)
# plt.tight_layout()
# plt.savefig('Path Analysis_Stud init.png'.format(units), dpi=300)
# plt.show()


# # %% Connectivity Histogram
# import seaborn as sns
# import pandas as pd
#
# k_in = [lay_1[i].sum(axis=0).flatten() for i in range(len(lay_1))]
# # concatenate the list of arrays into a single array
# k_in = np.concatenate(k_in)
# # normalize k_in values between 0 and 1
#
# lay_1_teacher = teacher.layers[1].base.numpy()
# k_in_teacher = lay_1_teacher.sum(axis=0).flatten()
#
# k_in = (k_in) / k_in.ptp()
# k_in_teacher = (k_in_teacher) / k_in_teacher.ptp()
#
# k_in_teacher = pd.DataFrame({'k_in': k_in_teacher, 'type': 'teacher'})
# k_in = pd.DataFrame({'k_in': k_in, 'type': 'student'})
# k_in = pd.concat([k_in, k_in_teacher])
# k_in = k_in.reset_index(drop=True)
#
# histogram_teacher = sns.histplot(k_in,
#                                  x='k_in',
#                                  common_norm=False,
#                                  hue='type',
#                                  shrink=.8,
#                                  stat='percent',
#                                  multiple='dodge',
#                                  binwidth=0.1
#                                  )
# plt.show()
# plt.close()
