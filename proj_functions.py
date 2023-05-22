import os
import shutil
import uuid
from os.path import join, dirname, abspath
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from math import floor, exp
import yaml
from ray.tune.progress_reporter import CLIReporter, Trial
import fnmatch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.config import list_physical_devices, set_visible_devices, experimental

physical_devices = experimental.list_physical_devices('GPU')
for dev in physical_devices:
    set_visible_devices(dev, 'GPU')
    experimental.set_memory_growth(dev, True)
import tensorflow
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.initializers import Initializer
from typing import Callable
from customspectral import Spectral
import numpy as np
from ray.tune.integration.keras import TuneReportCallback
from ray.tune import report


def LinearTeacher(config: dict) -> Callable:
    np.random.seed(0)
    a = np.sort(np.random.uniform(0, config['input_dim'], size=config['input_dim']))
    return lambda x: np.inner(a, x)


def DictionaryReplace(dictionary: dict, sobstitute: dict) -> dict:
    """
    Sostituisce le chiavi di un dizionario con i valori di un altro dizionario. La sostituizione avviene in profondità
    e solo se la chiave è uguale
    :param dictionary: Il primo dizionario in cui la sostituzione deve avvenire in accordo alle chiavi di sobstitute
    :param sobstitute: Dizionario con le chiavi da sostituire
    :return: Il dizionario con i valori sostituiti in accordo alle chiavi di sobstitute
    """

    for key, value in sobstitute.items():
        for original_key, original_value in dictionary.items():
            if isinstance(original_value, dict):
                dictionary[original_key] = DictionaryReplace(original_value, sobstitute)
            elif original_key == key:
                dictionary[original_key] = value
    return dictionary


def LoadConfig(config_name, hyper=None):
    """
    This function loads the configuration file from the config folder. If hyper is not None, it will replace the
    hyperparameters in the configuration file with the ones in hyper
    :param config_name: The name of the configuration file
    :param hyper: The dictionary with the hyperparameters to replace in the configuration file
    :return: The configuration dictionary
    """
    configuration_file = find(config_name, join(dirname(abspath(__file__))))[0]
    with open(configuration_file, 'r') as c:
        configuration = yaml.safe_load(c)

    if hyper is not None:
        return DictionaryReplace(configuration, hyper)
    else:
        return configuration


def NNTeacher(config: dict) -> Callable:
    """
    This function returns a neural network teacher with the configuration specified in config.
    :param config: The 'Dataset' subdictionary of the configuration file
    :return: The neural network teacher
    """

    first_hidden_dim = config.get('teacher_hidden', config['teacher_hidden'])
    second_hidden_dim = config.get('teacher_second', config['teacher_second'])

    np.random.seed(0)
    input_lay = Input(shape=config['input_dim'])

    first_layer_config = dict(units=first_hidden_dim,
                              activation='relu',
                              name='Hidden1',
                              diag_end_initializer='ones',
                              is_diag_end_trainable=False,
                              base_initializer=tensorflow.keras.initializers.glorot_uniform(seed=0),
                              use_bias=False,
                              trainable=False)

    second_layer_config = dict(units=second_hidden_dim,
                               activation='relu',
                               name='Hidden2',
                               diag_end_initializer='ones',
                               is_diag_end_trainable=False,
                               base_initializer=tensorflow.keras.initializers.glorot_uniform(seed=1),
                               use_bias=False,
                               trainable=False)

    y = Spectral(**first_layer_config)(input_lay)

    y1 = Spectral(**second_layer_config)(y)

    outputs = Dense(1,
                    bias_initializer='zeros',
                    activation='linear',
                    name='FinalOnes',
                    kernel_initializer='ones',
                    use_bias=False,
                    trainable=False)(y1)

    model = Model(inputs=input_lay,
                  outputs=outputs)

    return model


def MSERamp(trained, config):
    """
    This function computes the MSE ramp of a trained model in order to show the effect of the progressive network slimming
    and reveal the phase transition behaviour
    :param trained: The trained model
    :param config: The configuration dictionary
    :return:
    """
    model = clone_model(trained)
    model.compile(optimizer='Adam',
                  loss='MSE',
                  metrics=['MSE'])
    model.set_weights(trained.get_weights())
    base_lamb = RescaledEigenvalue(model)
    base_lamb = base_lamb / base_lamb.max()

    observable = []
    order_parameter = []
    x_test, y_test = DatasetMaker(config['Dataset'],
                                  return_test=True)
    trained.evaluate(x_test, y_test)
    model.evaluate(x_test, y_test)
    MSE_pre = ((model(x_test) - y_test) ** 2).numpy().sum() / y_test.shape[0]
    T = config['Dataset']['teacher_hidden']

    for cutoff in np.arange(0, 1, 0.05).tolist():
        index = base_lamb > cutoff
        # extract the MSE
        model.layers[1].diag_end.assign(
            trained.layers[1].diag_end.numpy() * index.reshape([1, model.layers[1].units]) * 1)
        MSE_post = ((model(x_test) - y_test) ** 2).numpy().sum() / x_test.shape[0]

        observable.append(abs(MSE_post - MSE_pre))
        order_parameter.append((index.sum() - T))

    return observable, order_parameter


def sigmoid(x): return 1 / (1 + exp(-x))


def DatasetMaker(config: dict, return_test=False, custom_function=None) -> tuple:
    """
    This function creates a dataset of patterns and labels based on the configuration dictionary given.
    Pass the configuration['Dataset'] dictionary to this function.
    :param config: configuration dictionary
    :param return_test: if True, returns a dataset of 10000 patterns
    :return:
    """
    if return_test:
        X = np.random.normal(0, 1, [10000, config['input_dim']])
    else:
        X = np.random.normal(0, 1, [config['patterns'], config['input_dim']])

    teacher_type = config['teacher']

    if custom_function is not None:
        Y = custom_function(X)
        return X, Y

    if teacher_type == 'linear':
        Y = LinearTeacher(config)(X)

    elif teacher_type == 'neural_network' or teacher_type == 'neural_network_hard':
        Y = NNTeacher(config)(X).numpy()

    else:
        raise ValueError('Teacher not available')

    return X, Y


class CustomTensorInit(Initializer):
    """
    Custom Initializer which returns the tensor given as argument
    """

    def __init__(self,
                 tensor):
        self.tensor_value = tensor

    def __call__(self, *args, **kwargs):
        return self.tensor_value

    def get_config(self):
        return {'tensor': self.tensor_value}


get_custom_objects().update({'CustomTensorInit': CustomTensorInit})


def ModelConstructor(config: dict, layer_type=None, custom_init=None, not_compiled=False):
    """
    This function creates a model based on the configuration dictionary given. The whole dictionary is needed
    :param not_compiled: if True, returns the uncompiled model
    :param config: configuration dictionary (whole, the one derived from the yaml file)
    :param layer_type: 'Direct' or 'Spectral'
    :param custom_init: Tensor to initialize the layer with, more specifically the base matrix
    :return: The tensorflow functional model
    """

    if custom_init is not None:
        kernel_init = CustomTensorInit(custom_init)
    else:
        kernel_init = 'GlorotUniform'

    # Functional Model definition
    input_lay = Input(shape=config['Dataset']['input_dim'])

    y = Spectral(**config['Student Layer 1']['Common'],
                 **config['Student Layer 1'][layer_type],
                 base_initializer=kernel_init)(input_lay)

    y = Dense(config['Dataset']['teacher_second'],
              trainable=True,
              activation='relu',
              name='Hidden2',
              kernel_initializer='GlorotUniform',
              bias_initializer='zeros',
              use_bias=False)(y)

    if config['Dataset']['teacher'] == 'hidden_manifold':
        last_activation = 'sigmoid'
    else:
        last_activation = 'linear'

    outputs = Dense(1,
                    bias_initializer='zeros',
                    activation=last_activation,
                    name='FinalOnes',
                    kernel_initializer='ones',
                    use_bias=False,
                    trainable=False)(y)

    model = Model(inputs=input_lay,
                  outputs=outputs)

    if not_compiled:
        return model

    model.compile(**config['Compile']['Common'],
                  optimizer=Adam(learning_rate=config['Compile'][layer_type]['learning_rate']))

    return model


def HyperparameterTuner(config: dict, trial_info: dict) -> None:
    """
    This function is used to tune the hyperparameters of the network. It is called by the Tune function.
    :param config: configuration dictionary containing the hyperparameters to be tuned
    :param trial_info: dictionary containing the information about the trial
    :return: None
    """
    configuration = LoadConfig(trial_info['configuration_name'], hyper=config)

    x_train, y_train = DatasetMaker(configuration['Dataset'])

    model = ModelConstructor(configuration, layer_type=trial_info['layer_type'])

    test_x, test_y = DatasetMaker(configuration['Dataset'],
                                  return_test=True)

    fit_results = model.fit(x_train, y_train,
                            validation_data=(test_x, test_y),
                            **configuration['Training']['Common'],
                            **configuration['Training'][trial_info['layer_type']],
                            callbacks=[TuneReportCallback({"val_Accuracy": "val_Accuracy"})],
                            verbose=0)

    return {"val_Accuracy": fit_results['val_Accuracy'][-1]}


def TrainModel(config: dict) -> None:
    """
    This function trains the model and saves the results in the results folder. It is called by the main function.
    :param config: It is a dictionary containing the configuration of the model
    """
    x_train, y_train = DatasetMaker(config)

    model = ModelConstructor(config)

    print('Start Fit...')
    model.fit(x_train, y_train,
              batch_size=500,
              epochs=config['epochs'],
              verbose=0)

    print('Start Test...')
    test_x, test_y = DatasetMaker(config,
                                  return_test=True)
    print('Train Loss')
    train_output = model.evaluate(x_train, y_train,
                                  batch_size=1000,
                                  return_dict=True)
    print('Test Loss')
    test_output = model.evaluate(test_x, test_y,
                                 batch_size=1000,
                                 return_dict=True)

    if config['debug']:
        model.evaluate(test_x, test_y,
                       batch_size=500)

    results_dict = dict(test_loss=test_output['loss'],
                        train_loss=train_output['loss'])

    if not config['return_model']:
        report(**results_dict)
        ResultsHandler(config).SaveResults(model, results_dictionary=results_dict)

    return model if config['return_model'] else None


class ResultsHandler:
    """
    This class is used to save the results of the experiments. Takes as input the configuration dictionary.
    It extracts the results folder path and the name of the experiment from it and create the folder if it does not exist.
    The class has a method to save the results of the experiment in a csv file from a dictionary passed as arguments.
    plus the column 'parameters_filename' where there is the name of the file containing the parameters of the base matrix and eigenvalues saved
    as npz with a unique name in the Parameter folder (created if it does not exist).
    """

    def __init__(self, config: dict):
        current_path = os.path.dirname(os.path.abspath(__file__))

        # Check if the configuration dictionary contains the results folder path and the name of the experiment. raise error if not
        if 'results_folder' not in config['Global'].keys():
            raise ValueError('The configuration dictionary does not contain the results folder path (results_folder)')
        if 'experiment_name' not in config['Global'].keys():
            raise ValueError(
                'The configuration dictionary does not contain the name of the experiment (experiment_name)')

        self.results_folder = config['Global']['results_folder']
        self.experiment_name = config['Global']['experiment_name']
        self.results_path = os.path.join(current_path, 'Results', self.results_folder, self.experiment_name)
        self.parameters_path = os.path.join(self.results_path, 'Parameters')
        if 'Save' in config.keys():
            self.models_path = os.path.join(self.results_path, 'Models')
        else:
            self.models_path = None
        self.font_style = dict(fontsize=15, fontdict={'fontname': 'Serif'})
        # Create the experiment folder if it does not exist
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def SaveResults(self, model: Model, results_dictionary: dict, with_parameters: bool = True):
        """
        This function saves the results of the experiment in a csv file from a dictionary passed as arguments.
        If with_parameters is True, adds a column 'parameters_filename' where there is the name of the file containing
        the parameters of the base matrix and eigenvalues saved. This is created with a unique name and stored in the
        Parameter folder (created if it does not exist).
        :param model: Keras model
        :param results_dictionary: dictionary containing the results of the experiment
        :param with_parameters: boolean indicating if the parameters of the base matrix and eigenvalues should be saved
        :return:
        """
        if with_parameters:
            if not os.path.exists(self.parameters_path):
                os.makedirs(self.parameters_path)

            parameters_filename = 'Parameters_' + str(uuid.uuid4()) + '.npz'
            base = model.layers[1].base.numpy()
            diag_end = model.layers[1].diag_end.numpy()
            np.savez(os.path.join(self.parameters_path, parameters_filename),
                     base=base,
                     diag_end=diag_end,
                     eigenvalues=RescaledEigenvalue(model),
                     second_layer=model.layers[2].kernel.numpy(),
                     savez_compressed=True)
            if self.models_path is not None:
                if not os.path.exists(self.models_path):
                    os.makedirs(self.models_path)

                model_filename = parameters_filename.replace('.npz', '.h5')
                model_filename = model_filename.replace('Parameters_', 'Model_')

                model.save(os.path.join(self.models_path, model_filename))

            results_dictionary['parameters_filename'] = parameters_filename
            results_dictionary['model_filename'] = model_filename

        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.DataFrame(results_dictionary, index=[0])
        if os.path.exists(results_filename):
            df.to_csv(results_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(results_filename, index=False)
        return None

    def ReturnAggregatedResults(self,
                                conditions_dictionary: dict,
                                also_csv: bool = False) -> dict | pd.DataFrame:
        """
        This method returns the aggregated parameters of the experiment. It takes as input a dictionary containing the conditions to aggregate the results.
        The Results.csv file is read and the aggregated parameters are returned according to the conditions. The parameters
        are extracted from the path indicated in parameters_filename if present in the results csv file. Raise an error
        otherwise.
        :param also_csv: If True, returns also the dataframe containing the aggregated results
        :param conditions_dictionary: the dictionary containing the conditions to aggregate the results. The keys are the columns of the Results.csv file.
        :return: a dictionary containing the aggregated parameters of the experiment, namely base and eigenvalues. Optionally,
        it returns also the dataframe containing the aggregated results.
        """
        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)

        for key, value in conditions_dictionary.items():
            df = df[df[key] == value]

        if 'parameters_filename' in df.columns:
            base_list = []
            eigenvalues_list = []
            for parameters_filename in df['parameters_filename']:
                parameters = np.load(os.path.join(self.parameters_path, parameters_filename))
                base_list.append(parameters['base'])
                eigenvalues_list.append(parameters['eigenvalues'])
            if also_csv:
                return {'base': np.array(base_list), 'eigenvalues': np.array(eigenvalues_list)}, df
            else:
                return {'base': np.array(base_list), 'eigenvalues': np.array(eigenvalues_list)}
        else:
            raise ValueError('The Results.csv file does not contain the column parameters_filename')

    def LossComparePlot(self, x_axis: str = 'units', conditions_dictionary: dict = None, save: bool = False):
        """
        This method plots the loss of the experiment. It takes as input a dictionary containing the conditions to aggregate the results
        and the x_axis to plot. The Results.csv file is read and the train and test losses are plotted in two different panels. The plots are done using seaborn library with errorbars.
        In each plot the color are different basing on the layer type. If the conditions dictionary is not provided, the plot is done on the whole dataset.
        """
        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)

        if conditions_dictionary is not None:
            for key, value in conditions_dictionary.items():
                df = df[df[key] == value]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        # set seaborn font style to serif
        sb.lineplot(data=df, x=x_axis, y='train_loss', hue='layer_type', ax=ax1, errorbar=('sd', 2))
        sb.lineplot(data=df, x=x_axis, y='test_loss', hue='layer_type', ax=ax2, errorbar=('sd', 2))
        # set y axis label
        ax1.set_ylabel('Train Loss', **self.font_style)
        ax2.set_ylabel('Test Loss', **self.font_style)
        # set x axis label
        ax1.set_xlabel(x_axis, **self.font_style)
        ax2.set_xlabel(x_axis, **self.font_style)
        # set tick label size
        ax1.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        # remove legend title, set legend font size and set legend font to serif
        ax1.legend(title=None, fontsize=15, prop={'family': 'serif'})
        ax2.legend(title=None, fontsize=15, prop={'family': 'serif'})
        # set title of the figure
        # log scale for the y axis
        ax1.set_yscale('log')
        ax2.set_yscale('log')

        fig.suptitle('Losses with respect to {}'.format(x_axis), fontsize=18, fontdict={'fontname': 'Serif'})
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.results_path, 'Losses.png'), dpi=600)
        plt.show()
        return df

    def EigenvaluesHistogram(self,
                             conditions_dictionary: dict = None,
                             common_extrema: bool = True,
                             save: bool = False,
                             return_zero_percentage: bool = False,
                             return_data=False,
                             debug=False,
                             **kwargs):
        """
        This method plots the histogram of the eigenvalues of the experiment. It takes as input a dictionary containing the conditions to aggregate the results.
        The Results.csv file is read and the histogram of the eigenvalues is plotted extracting the results from the parameters_filename column.
        The plots are done using seaborn library. If the conditions dictionary is not provided, the plot is done on the whole dataset. """

        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)

        if conditions_dictionary is not None:
            for key, value in conditions_dictionary.items():
                df = df[df[key] == value]

        # Represent in the same plot different histograms (one for each layer type) extracting the eigenvalues from the parameters_filename column of
        # the Results.csv file.

        to_plot_dictionary = {}
        stat_dictionary = {}
        for lay in df['layer_type'].unique():
            eigenvalues = []
            sub_df = df[df['layer_type'] == lay]
            eig_list = []
            for parameters_filename in sub_df['parameters_filename']:
                parameters = np.load(join(self.parameters_path, parameters_filename))
                eigenvalues.append(parameters['eigenvalues'])

            per_trial = (abs(np.array(eigenvalues)) / abs(np.array(eigenvalues)).max())[:, 0, :]
            eigenvalues = abs(np.array(eigenvalues).flatten())

            # if common_extrema is True, eigenvalues are normalized to the same extrema: [0, 1]. The bars are side by side.
            # if common_extrema is False, eigenvalues are not normalized and the bars are stacked.
            if common_extrema:
                eigenvalues = eigenvalues / (eigenvalues.max())
            to_plot_dictionary[lay] = eigenvalues
            stat_dictionary[lay] = per_trial

        print('Eigenvalues are normalized to the same extrema: [0, 1]')
        # create a melted dataframe to plot the histograms
        df_melted = pd.melt(pd.DataFrame(to_plot_dictionary))
        # plot the histograms with rescaled bars (side by side) by a factor of 0.9
        # if ax is passed plot in the given axis
        plot_argument = dict(data=df_melted,
                             x='value',
                             ax=kwargs.get('ax'),
                             hue='variable',
                             stat='percent',
                             shrink=.95,
                             common_norm=False,
                             binwidth=0.05,
                             multiple='dodge',
                             alpha=.9)

        if kwargs.get('ax') is not None:
            plot_function = sb.histplot
        else:
            plot_function = sb.displot
            # remove ax from plot_argument
            plot_argument.pop('ax')
            plot_argument['kind'] = 'hist'

        if return_data:
            return plot_argument['data']
        else:
            plot_function(**plot_argument)
        # remove legend title, set legend font size and set legend font to serif in the given axis
        if kwargs.get('ax') is None:
            plt.legend(title=None, fontsize=15, prop={'family': 'serif'})
        else:
            kwargs.get('ax').legend(fontsize=13, prop={'family': 'serif'})

        # set x,y label font size and font to serif in the given axis
        if kwargs.get('ax') is None:
            plt.xlabel('Eigenvalues', **self.font_style)
            plt.ylabel('Percentage', **self.font_style)
            plt.tick_params(labelsize=13)
            plt.title('Eigenvalues Histogram', **self.font_style)
        else:
            kwargs.get('ax').set_xlabel('Normalized $\mathcal{W},\mathcal{L}$', **self.font_style)
            kwargs.get('ax').set_ylabel('Percentage', **self.font_style)
            kwargs.get('ax').tick_params(labelsize=13)

        # set title of the figure
        conditions_string = ''
        for key, value in conditions_dictionary.items():
            conditions_string += '{}: {}'.format(key, value)
        if kwargs.get('ax') is None:
            plt.title(conditions_string, **self.font_style)
        else:
            kwargs.get('ax').set_title(conditions_string, **self.font_style)
        if save:
            # save the plot with a name that contains the conditions
            conditions_string = ''
            for key, value in conditions_dictionary.items():
                conditions_string += '{}_{}_'.format(key, value)
            # plt.savefig(os.path.join(self.results_path, 'EigenvaluesHistogram_{}.png'.format(conditions_string)))
            # Return the figure in order to plot a grid of histograms later
            if return_zero_percentage:
                plt.close()
                if kwargs.get('ax') is None and not return_zero_percentage:
                    plt.show()

        if return_zero_percentage:
            # Save percent of nonzero eigenvalues (smaller than binwidth) with mean and variance
            nonzero_eigenvalues = {}
            units = conditions_dictionary.get('units')
            if units is None:
                raise ValueError('Units must be specified in the conditions dictionary')
            for lay in df['layer_type'].unique():
                not_zero = np.sum(1 - (stat_dictionary[lay] < 0.05) * 1, axis=1)
                nonzero_eigenvalues[lay] = {'mean': not_zero.mean(),
                                            'std': not_zero.std()}
            return nonzero_eigenvalues

    def RemoveResults(self, unit_list):
        """ This method removes the results and parameters of the experiments with the specified units. The parameters
        are removed according to the parameters_filename column of the Results.csv file."""

        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)
        for units in unit_list:
            df = df[df['units'] != units]
        df.to_csv(results_filename, index=False)
        # remove the parameters files
        for units in unit_list:
            for parameters_filename in df[df['units'] == units]['parameters_filename']:
                os.remove(join(self.parameters_path, parameters_filename))

    def AverageDimension(self,
                         y_line: float = 20,
                         save: bool = False, ):
        """ This method plots the average dimension of the network as a function of the number of units in the hidden. The
        is computed using the non zero eigenvalues related scalars of the network: The eigenvalues multiplied by the L_2
        norm of the eigenvector row (for the 'Spectral') or the feature norm for the conventional training ('Direct')."""

        results_filename = os.path.join(self.results_path, 'Results.csv')
        datas = pd.read_csv(results_filename)

        df = pd.DataFrame()

        # collect all the results in a dataframe
        lista = datas['units'].unique().tolist()
        for units in lista:
            conditions = {'units': units}
            non_zero_eigenvalues = self.EigenvaluesHistogram(conditions_dictionary=conditions,
                                                             return_zero_percentage=True)
            for key, value in non_zero_eigenvalues.items():
                new_data = pd.DataFrame({'units': [units],
                                         'layer_type': [key],
                                         'mean': [value['mean']],
                                         'error': [value['std']]})
                df = pd.concat([df, new_data], ignore_index=True)

        # plot the results using a shaded error bar without seabron andusing 'fill inbetween' by matplotlib

        df_spectral = df[df['layer_type'] == 'Spectral']
        df_direct = df[df['layer_type'] == 'Direct']

        # sort the dataframes by units
        df_spectral = df_spectral.sort_values(by='units')
        df_direct = df_direct.sort_values(by='units')

        # Plotting
        plt.figure(figsize=(7, 4), dpi=200)

        # Plot Spectral data with shaded region
        plt.plot(df_spectral['units'], df_spectral['mean'], marker='o', label='Spectral')
        plt.fill_between(df_spectral['units'],
                         df_spectral['mean'] - df_spectral['error'],
                         df_spectral['mean'] + df_spectral['error'],
                         alpha=0.3)

        # Plot Direct data with shaded region
        plt.plot(df_direct['units'], df_direct['mean'], marker='o', label='Direct')
        plt.fill_between(df_direct['units'],
                         df_direct['mean'] - df_direct['error'],
                         df_direct['mean'] + df_direct['error'],
                         alpha=0.3)

        plt.xlabel('Units', **self.font_style)
        plt.ylabel('Average Dimension', **self.font_style)
        # set y log scale
        plt.tick_params(labelsize=13)
        # add horizontal line at NN teacher dimension
        plt.axhline(y=y_line, color='black', linestyle='--', label='Teacher Size')
        plt.legend(title=None, fontsize=15, prop={'family': 'serif'})
        plt.title('Average Dimension', **self.font_style)
        plt.tight_layout()
        print(df)
        if save:
            plt.savefig(os.path.join(self.results_path, 'AverageDimension.png'))
        plt.show()

    def ExtractParameters(self, condition_dictionary):
        """ This method extracts all the parameters of the experiments with the specified units. The parameters
        are extracted according to the parameters_filename column of the Results.csv file."""

        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)
        parameter_list = []
        if condition_dictionary.get('epochs') is None:
            fnames = df[(df['units'] == condition_dictionary['units']) & (
                    df['layer_type'] == condition_dictionary['layer_type'])]['parameters_filename']
        else:
            fnames = df[(df['units'] == condition_dictionary['units']) & (
                    df['layer_type'] == condition_dictionary['layer_type']) & (
                                df['epochs'] == condition_dictionary['epochs'])]['parameters_filename']

        for parameters_filename in fnames:
            npz = np.load(join(self.parameters_path, parameters_filename))
            # extract the dictionary from npz
            parameters_dictionary = {key: npz[key] for key in npz.files}
            parameter_list.append(parameters_dictionary)
        return parameter_list

    def ExtractModels(self, condition_dictionary):
        """ This method extracts all the models of the experiments with the specified units. The models
        are extracted according to the parameters_filename column of the Results.csv file."""

        results_filename = os.path.join(self.results_path, 'Results.csv')
        df = pd.read_csv(results_filename)
        model_list = []
        fnames = df[(df['units'] == condition_dictionary['units']) & (
                df['layer_type'] == condition_dictionary['layer_type'])]['model_filename']
        for model_filename in fnames:
            model = load_model(join(self.models_path, model_filename), custom_objects={'Spectral': Spectral})
            model_list.append(model)
        return model_list


def ModelFitEval(model: Model, config: dict, layer_type='Spectral', print_results=False):
    """
    This function fits the model and evaluates the train and test loss.
    :param model: The tensorflow model to fit
    :param config: The configuration dictionary extracted from a yaml file, for this project it is the config.yaml file
    :param layer_type: 'Spectral' or 'Direct', depending on the layer type to fit
    :param print_results: If True, the train and test loss are printed for each epoch
    :return: a tuple (model, results_dictionary) where model is the fitted model and results_dictionary is a dictionary
    containing the test and train loss and the history of the training
    """
    x_train, y_train = DatasetMaker(config['Dataset'])
    test_x, test_y = DatasetMaker(config['Dataset'],
                                  return_test=True)

    h = model.fit(x_train,
                  y_train,
                  validation_split=0.2,
                  **config['Training']['Common'],
                  **config['Training'][layer_type],
                  verbose=0)

    train_loss = ((model(x_train) - y_train) ** 2).numpy().sum() / x_train.shape[0]
    test_loss = ((model(test_x) - test_y) ** 2).numpy().sum() / test_x.shape[0]

    if print_results:
        print('Evaluating MSE...')
        print('Train Loss: {:.5f}'.format(train_loss))
        print('Test Loss: {:.5f}'.format(test_loss))

    return model, dict(test_loss=test_loss,
                       history=h,
                       train_loss=train_loss,
                       layer_type=layer_type)


def SpectralPrune(model, percentile):
    """
    This function takes a model and evaluates the percentile 'percentile' of the eigenvalues of every spectral layer
    present. Then it sets to 0 every diag_end element of the spectral layer whose magnitude is below the percentile.
    :param model:
    :param percentile:
    :return:
    """
    eigenvalues = np.array([])
    for layer in model.layers:
        if isinstance(layer, Spectral):
            # get the diag_end and append concat to eigenvalues of every layer
            eigenvalues = np.concatenate((eigenvalues, layer.diag_end.numpy()))
    # get the percentile
    percentile_value = np.percentile(np.abs(eigenvalues), percentile)
    for layer in model.layers:
        if isinstance(layer, Spectral):
            # set to 0 the elements of the diag_end whose magnitude is below the percentile
            layer.diag_end.assign(layer.diag_end * (np.abs(layer.diag_end) > percentile_value))
    return model


def OldResultsRemover(trial_info: dict):
    """
    Removes the old results of the experiment by deleting the folder containing the results
    :param trial_info:
    :return:
    """
    if isinstance(trial_info, str):
        config = LoadConfig(trial_info)
    else:
        config = LoadConfig(trial_info['configuration_name'])

    current_path = os.path.dirname(os.path.abspath(__file__))
    results_folder = config['Global']['results_folder']
    experiment_name = config['Global']['experiment_name']

    results_path = os.path.join(current_path, 'Results', results_folder, experiment_name)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
        print('Old results removed')
    else:
        print('No old results to remove')


class TrialTerminationReporter(CLIReporter):
    """
    Tune reporter that reports only on trial termination events.
    """
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def SpectralDirectSameInit(trial_info: dict | str,
                           plot_loss: bool = False,
                           save_results: bool = False,
                           return_model: bool = False,
                           return_results: bool = False,
                           only_spectral: bool = False,
                           ) -> dict:
    """
    Returns the results of the spectral and direct model trained on the same dataset and from the same initialization
    in the dictionary format

    :param return_results: If True returns also the results of the two training in the dictionary format with or without the
    trained models, depending on the value of the parameter return_model.
    IMPORTANT: Needs to be false if the function is
    running inside a tune.run() function. Tensorflow errors will occur otherwise.

    :param return_model: If True returns also the trained models (only if return_results is True)

    :param save_results: If True, saves the results in the Results.csv file using the class ResultsSaver

    :param trial_info: Either a string containing the name of the configuration file or a dictionary with the keys
    'configuration_name' and 'hyperparameters'. If also 'return_model' is True, the function returns the two trained
    models

    :param plot_loss: if True, plots the loss of the two models with respect to the epochs

    :return: if return_model is True, returns the two models, otherwise, if return_results is True, returns the results of the two
    training in the dictionary format.
    Dictionary output format:
    {
        'model_spectral': trained_spectral,
        'model_direct': trained_direct,
        'spectral_results': spectral_results,
        'direct_results': direct_results
    }

    """

    if return_model and not return_results:
        raise ValueError('return_model can be True only if return_results is True')

    if isinstance(trial_info, str):
        config = LoadConfig(trial_info)
    else:
        if trial_info.get('hyperparameters') is not None:
            config = LoadConfig(trial_info['configuration_name'], hyper=trial_info['hyperparameters'])
        else:
            config = LoadConfig(trial_info['configuration_name'])

    model_spectral = ModelConstructor(config,
                                      layer_type='Spectral')

    model_direct = ModelConstructor(config,
                                    custom_init=model_spectral.layers[1].base.numpy(),
                                    layer_type='Direct')

    if (model_spectral.layers[1].base.numpy() != model_direct.layers[1].base.numpy()).all():
        raise ValueError('The two models have different initializations')

    if trial_info.get('hyperparameters') is not None:
        print_loss = False
    else:
        print_loss = True

    if trial_info.get('hyperparameters') is None:
        print('Start Fit Spectral...')

    trained_spectral, spectral_results = ModelFitEval(model_spectral, config, layer_type='Spectral',
                                                      print_results=print_loss)

    if trial_info.get('hyperparameters') is None:
        print('Start Fit Direct...')

    if not only_spectral:
        trained_direct, direct_results = ModelFitEval(model_direct, config, layer_type='Direct',
                                                      print_results=print_loss)
    else:
        trained_direct = None
        direct_results = None
    if plot_loss:
        if config['Compile']['Common']['loss'] == 'categorical_crossentropy':
            loss_name = 'Accuracy'
        else:
            loss_name = 'MSE'

        plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
        plt.subplot(1, 2, 1)
        plt.plot(spectral_results['history'].history[loss_name], label='Spectral')
        if not only_spectral:
            plt.plot(direct_results['history'].history[loss_name], label='Direct')
        plt.yscale('log')
        plt.title('Train ' + loss_name)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(spectral_results['history'].history['val_' + loss_name], label='Spectral')
        if not only_spectral:
            plt.plot(direct_results['history'].history['val_' + loss_name], label='Direct')
        plt.title('Test ' + loss_name)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if save_results:
        if trial_info.get('hyperparameters') is not None:
            spectral_results = {**spectral_results, **trial_info['hyperparameters']}
            if not only_spectral:
                direct_results = {**direct_results, **trial_info['hyperparameters']}
        print('Saving results...')
        if not only_spectral:
            ResultsHandler(config).SaveResults(trained_direct, results_dictionary=direct_results)
        ResultsHandler(config).SaveResults(trained_spectral, results_dictionary=spectral_results)

    if return_results:
        if return_model:
            return {'model_spectral': trained_spectral,
                    'model_direct': trained_direct,
                    'spectral_results': spectral_results,
                    'direct_results': direct_results}
        else:
            return {'spectral_results': spectral_results,
                    'direct_results': direct_results}


def SetEigenvaluesToZero(model: Model, perc: int):
    """
    Sets to zero the eigenvalues whose magnitude is below the perc percentile and retur the modified model (in place)
    :param model: the model to be modified
    :param perc: the percentile
    :return: the modified model
    """
    rescaled_eig = RescaledEigenvalue(model)
    eigen = model.layers[1].diag_end.numpy()
    threshold = np.percentile(np.abs(rescaled_eig), perc)
    eigen[np.abs(rescaled_eig) < threshold] = 0
    model.layers[1].diag_end.assign(eigen)
    return model


def RescaledEigenvalue(model: Model):
    """
    Returns the rescaled eigenvalue of the spectral layer of the model
    :param model: The model to be evaluated and whose spectral layer eigenvalue is to be returned
    :return:
    """
    base = model.layers[1].base.numpy()
    rescaled_eig = abs(model.layers[1].diag_end.numpy()) * np.linalg.norm(base, axis=0)
    return rescaled_eig * (model.layers[1].get_config().get('units') ** 0.5)


def FlattenList(l: list):
    """
    Flattens a list of lists
    :param l: The list to be flattened
    :return: The single flattened list
    """
    return [item for sublist in l for item in sublist]


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
