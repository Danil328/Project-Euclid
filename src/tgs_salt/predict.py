import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import make_model
from helpers import read_by_pyvips
from scipy.misc import imresize
from skimage.util import pad
from tqdm import tqdm


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def predict():
    model = make_model(
        network='vanilla_unet',
        input_shape=(284, 284, 3),
        with_sigmoid=True,
        random_state=17
    )
    model.load_weights('../../models/best_standard_vanilla_unet_fold_4.h5')

    pred_dict = dict()
    test_image_names = os.listdir('../../data/source/test/images')
    for test_image_name in tqdm(test_image_names):
        image = read_by_pyvips(os.path.join('../../data/source/test/images', test_image_name))
        image = imresize(image, (100, 100), interp='nearest')
        image_pad = pad(
            image,
            ((92, 92), (92, 92), (0, 0)),
            mode='reflect'
        )
        image_pad = image_pad / 255.
        predictions = model.predict(np.expand_dims(image_pad, axis=0))
        predictions = predictions > 0.5
        predictions = predictions.astype(np.uint8)

        pred_dict[test_image_name.split('.')[0]] = RLenc(imresize(predictions[0, :, :, 0], (101, 101), interp='nearest'))

        # plt.imshow(image)
        # plt.show()
        #
        # plt.imshow(predictions[0, :, :, 0])
        # plt.show()


    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('../../predictions/submission.csv')

    a = 5












if __name__ == '__main__':
    predict()