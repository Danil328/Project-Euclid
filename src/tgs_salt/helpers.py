import os
import pyvips
import numpy as np


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

def read_train_data_to_memory(train_dir, channels):
    images_dict = read_train_data_to_memory_separately(
        os.path.join(train_dir, 'images'),
        channels
    )
    masks_dict = read_train_data_to_memory_separately(
        os.path.join(train_dir, 'masks'),
        channels
    )

    return images_dict, masks_dict

def read_train_data_to_memory_separately(data_dir, channels):
    data_dict = dict()
    file_names = os.listdir(data_dir)

    for file_name in file_names:
        absolute_file_pathway = os.path.join(data_dir, file_name)
        if data_dir[-6:] == 'images':
            data = read_by_pyvips(absolute_file_pathway)
        else:
            data = read_by_pyvips(absolute_file_pathway, grayscale=True)

        if channels == 3:
            data_dict[file_name.split('.')[0]] = data
        elif channels == 1:
            data_dict[file_name.split('.')[0]] = np.expand_dims(data[..., 0], axis=2)
        else:
            raise ValueError('Сhannel must be either 1 or 3')

    return data_dict

def take_image_names(dataframe, folds):
    dataframe = dataframe[dataframe['fold'].isin(folds)]
    image_names = dataframe['id'].tolist()

    return image_names