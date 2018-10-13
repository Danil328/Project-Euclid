import os
import pandas as pd
import numpy as np
from helpers import read_by_pyvips
from imgaug import augmenters as iaa

def get_augmentations():
    image_augmentations = iaa.Sequential([
        iaa.Fliplr(0.5, name='Fliplr'),
        iaa.Sometimes(0.5, iaa.Add((-40, 40)))
    ])

    mask_augmentations = iaa.Sequential([
        iaa.Fliplr(0.5, name='Fliplr'),
    ])

    return {'image_augmentations': image_augmentations, 'mask_augmentations': mask_augmentations}

def create_stratified_validation(n_folds, source_dir, stratified_by):
    train_df = pd.read_csv(os.path.join(source_dir, 'train.csv'))
    depths_df = pd.read_csv(os.path.join(source_dir, 'depths.csv'))
    depths_df = depths_df[
        depths_df['id'].isin(train_df['id'])
    ]
    depths_df['area'] = np.nan

    if stratified_by == 'depth':
        depths_df.sort_values('z', inplace=True)

    else:
        for index, row in depths_df.iterrows():
            image_name = row['id'] + '.png'
            mask_pathway = os.path.join(source_dir + '/train/masks', image_name)
            mask = read_by_pyvips(mask_pathway, grayscale=True)
            area = np.count_nonzero(mask) / np.power(mask.shape[0], 2)
            depths_df.set_value(
                index=index,
                col='area',
                value=np.round(area, 2)
            )

        if stratified_by == 'coverage':
            depths_df.sort_values('area', inplace=True)
        elif stratified_by == 'all':
            depths_df.sort_values(['area', 'z'], inplace=True)
        else:
            raise ValueError('There is no such mode, available: depth, area, all')

    depths_df['fold'] = (list(range(1, n_folds+1)) * depths_df.shape[0])[:depths_df.shape[0]]

    return depths_df[['id', 'fold']]

def get_train_folds(n_folds, test_mode, valid_fold):
    train_folds = list(range(1, n_folds+1))
    if test_mode:
        if valid_fold == train_folds[-1]:
            raise ValueError('Last fold reserved for the test')
        train_folds = train_folds[:-1]
    train_folds.remove(valid_fold)

    return train_folds