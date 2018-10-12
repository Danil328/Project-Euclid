import os
import pyvips
import numpy as np
from itertools import combinations
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard, CSVLogger


def read_by_pyvips(path, grayscale=False):
    image = pyvips.Image.new_from_file(path, access='sequential')
    if grayscale:
        image = image.colourspace('b-w')

    memory_image = image.write_to_memory()
    numpy_image = np.ndarray(
        buffer=memory_image,
        dtype=np.uint8,
        shape=[image.height, image.width, image.bands]
    )

    return numpy_image

def get_folds_iterations(n_folds):
    folds_iterations = list()
    folds = list(range(n_folds))
    train_folds_per_iteration = list(combinations(folds, n_folds-1))

    for train_folds in train_folds_per_iteration:
        valid_fold = list(set(folds) - set(train_folds))
        folds_iterations.append([
            train_folds,
            valid_fold
        ])

    return folds_iterations

def read_train_data_to_memory():
    images_dict = read_train_data_to_memory_separately('images')
    masks_dict = read_train_data_to_memory_separately('masks')

    return images_dict, masks_dict

def read_train_data_to_memory_separately(data_name):
    data_dict = dict()
    data_pathway = '../../data/source/train'
    file_names = os.listdir(os.path.join(data_pathway, data_name))

    for file_name in file_names:
        absolute_file_pathway = os.path.join(data_pathway, data_name, file_name)
        if data_name == 'images':
            data = read_by_pyvips(absolute_file_pathway)
        else:
            data = read_by_pyvips(absolute_file_pathway, grayscale=True)

        data_dict[file_name.split('.')[0]] = data

    return data_dict

def take_image_names(dataframe, folds):
    dataframe = dataframe[dataframe['fold'].isin(folds)]
    image_names = dataframe['id'].tolist()

    return image_names

def callbacks_factory(callbacks_list, model_maskname='unknown_model', monitor='val_loss'):
    callbacks = list()

    if 'best_model_checkpoint' in callbacks_list:
        best_model_filepath = '../../models/best_{0}.h5'.format(model_maskname)
        best_model_checkpoint = ModelCheckpoint(
            filepath=best_model_filepath,
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            period=1
        )
        callbacks.append(best_model_checkpoint)

    if 'early_stopping' in callbacks_list:
        early_stopping = EarlyStopping(
            monitor=monitor,
            min_delta=0,
            patience=10,
            verbose=1,
            mode='max'
        )
        callbacks.append(early_stopping)

    if 'tensorboard' in callbacks_list:
        tensorboard = TensorBoard(log_dir='../../logs/{0}'.format(model_maskname))
        callbacks.append(tensorboard)

    if 'csv_logger' in callbacks_list:
        csv_logger = CSVLogger(filename='../../logs/{0}.log'.format(model_maskname))
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