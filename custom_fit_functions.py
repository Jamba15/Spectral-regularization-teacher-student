from tensorflow.keras.optimizers import Adam
from proj_functions import LoadConfig, ModelConstructor, DatasetMaker
from tensorflow.keras.models import Model
from tensorflow import GradientTape
import tensorflow as tf
trial_info = {'configuration_name': 'alternatingL2.yml',
              'hyperparameters': {'units': 100,
                                  'epochs': 30},
              'return_model': True,
              'return_results': True}

if isinstance(trial_info, str):
    config = LoadConfig(trial_info)
else:
    if trial_info.get('hyperparameters') is not None:
        config = LoadConfig(trial_info['configuration_name'],
                            hyper=trial_info['hyperparameters'])
    else:
        config = LoadConfig(trial_info['configuration_name'])

model = ModelConstructor(config, layer_type='Spectral')


class AlternateTraining(Model):
    """
    This class is used to train the spectral layers in the model in two alternating phases. The first phase trains the
    eigenvalues and leaves the base fixed. The second phase trains the base and leaves the eigenvalues fixed. This is done
    by computing the gradient only with respect to one of the variables. The gradient is then applied to the variable.
    """

    def __init__(self, train_step_period=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_step_period = int(train_step_period)

        self.phase = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.change_mode = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.train_mode = tf.Variable('diag_end', dtype=tf.string, trainable=False)

        tf.print('Initialized training mode to', self.train_mode)
        tf.print('Initial phase is', self.phase)

    @tf.function
    def train_step(self, data):
        self.phase.assign_add(1)
        # Get arguments from higher level fit function
        # Unpack the data. Its structure depends on your model
        x, y = data

        with GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)
        diag_mode_index = [i for i in range(len(trainable_vars)) if 'base' not in trainable_vars[i].name]
        base_mode_index = [i for i in range(len(trainable_vars)) if 'diag_end' not in trainable_vars[i].name]

        tf.cond(tf.cast(self.train_mode == 'diag_end', tf.bool),
                lambda: self.optimizer.apply_gradients(zip([gradient[i] for i in diag_mode_index],
                                                           [trainable_vars[i] for i in diag_mode_index])),
                lambda: self.optimizer.apply_gradients(zip([gradient[i] for i in base_mode_index],
                                                           [trainable_vars[i] for i in base_mode_index]))
                )

        tf.cond(tf.cast(self.phase % self.train_step_period == 0, tf.bool),
                lambda: self.change_mode.assign(tf.logical_not(self.change_mode)),
                lambda: self.change_mode.assign(self.change_mode))

        tf.cond(self.change_mode,
                lambda: self.train_mode.assign('base'),
                lambda: self.train_mode.assign('diag_end'))

        # Compute metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


new_model = AlternateTraining(inputs=model.inputs, outputs=model.outputs, name='new_model', train_step_period=10)
new_model.compile(**config['Compile']['Common'],
                  optimizer=Adam(learning_rate=config['Compile']['Spectral']['learning_rate']))
x, y = DatasetMaker(config['Dataset'])
test_x, test_y = DatasetMaker(config['Dataset'], return_test=True)
new_model.fit(x, y, **config['Training']['Common'],
              **config['Training']['Spectral'],
              validation_data=(test_x, test_y))
