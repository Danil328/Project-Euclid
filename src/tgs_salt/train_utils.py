import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD


def callbacks_factory(callbacks_list, model_maskname, monitor, monitor_mode, early_stopping_parience, models_dir, logs_dir):
    callbacks = list()

    if 'best_model_checkpoint' in callbacks_list:
        best_model_pathway = os.path.join(models_dir, 'best_{0}.h5').format(model_maskname)
        best_model_checkpoint = ModelCheckpoint(
            filepath=best_model_pathway,
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode=monitor_mode,
            period=1
        )
        callbacks.append(best_model_checkpoint)

    if 'last_model_checkpoint' in callbacks_list:
        last_model_pathway = os.path.join(models_dir, 'last_{0}.h5').format(model_maskname)
        last_model_checkpoint = ModelCheckpoint(
            filepath=last_model_pathway,
            monitor=monitor,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode=monitor_mode,
            period=1
        )
        callbacks.append(last_model_checkpoint)

    if 'early_stopping' in callbacks_list:
        early_stopping = EarlyStopping(
            monitor=monitor,
            min_delta=0,
            patience=early_stopping_parience,
            verbose=1,
            mode=monitor_mode
        )
        callbacks.append(early_stopping)

    if 'tensorboard' in callbacks_list:
        tensorboard = TensorBoard(log_dir=os.path.join(logs_dir, '{0}').format(model_maskname))
        callbacks.append(tensorboard)

    if 'csv_logger' in callbacks_list:
        csv_logger = CSVLogger(filename=os.path.join(logs_dir,'{0}.log').format(model_maskname))
        callbacks.append(csv_logger)

    if 'learning_rate_scheduler' in callbacks_list:
        def exp_decay(epoch):
            initial_learning_rate = 0.001
            k = 0.1
            learning_rate = initial_learning_rate * np.exp(-k * epoch)

            return learning_rate

        callbacks.append(
            LearningRateScheduler(
                exp_decay,
                verbose=1
            )
        )

    return callbacks

def make_optimizer(optimizer_name, learning_rate, decay):

    if optimizer_name == 'rmsprop':
        return RMSprop(lr=learning_rate, decay=float(decay))

    elif optimizer_name == 'adam':
        return Adam(lr=learning_rate, decay=float(decay))

    elif optimizer_name == 'amsgrad':
        return Adam(lr=learning_rate, decay=float(decay), amsgrad=True)

    elif optimizer_name == 'nesterov':
        return SGD(lr=learning_rate, decay=float(decay), momentum=0.9, nesterov=True)

    else:
        raise NotImplementedError('Unknown optimizer')