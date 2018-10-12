import numpy as np
from keras.utils import Sequence
from scipy.misc import imresize
from skimage.util import pad
from sklearn.preprocessing import binarize


class DataGenerator(Sequence):
    def __init__(
            self,
            images_dict,
            masks_dict,
            image_names,
            augmentations=None,
            batch_size=32,
            shape=(101, 101),
            input_padding=None,
            shuffle=True):

        self.images_dict = images_dict
        self.masks_dict = masks_dict
        self.image_names = sorted(image_names)
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.shape = shape
        self.input_padding = input_padding
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_names) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image_names = self.image_names[index * self.batch_size: (index + 1) * self.batch_size]

        if self.input_padding:
            input_shape = (
                self.shape[0] + self.input_padding,
                self.shape[1] + self.input_padding
            )
        else:
            input_shape = self.shape

        batch_x = np.empty((self.batch_size, *input_shape, 3), dtype=np.uint8)
        batch_y = np.empty((self.batch_size, *self.shape, 1), dtype=np.uint8)

        for i, image_name in enumerate(batch_image_names):
            image = self.images_dict[image_name]
            mask = self.masks_dict[image_name][:, :, 0]
            image = imresize(image, self.shape, interp='nearest')
            mask = imresize(mask, self.shape, interp='nearest')

            if self.input_padding:
                one_side_padding = int(self.input_padding / 2)
                image = pad(
                    image,
                    ((one_side_padding, one_side_padding), (one_side_padding, one_side_padding), (0, 0)),
                    mode='reflect'
                )
            mask = binarize(mask, 255. / 2)

            batch_x[i, ...] = image
            batch_y[i, :, :, 0] = mask

        if self.augmentations:
            batch_x, batch_y = self._apply_augmentations(batch_x, batch_y)

        return batch_x / 255., batch_y

    def _apply_augmentations(self, batch_x, batch_y):
        image_augmentations = self.augmentations['image_augmentations']
        mask_augmentations = self.augmentations['mask_augmentations']

        image_augmentations.localize_random_state()

        image_augmentations_det = image_augmentations.to_deterministic()
        mask_augmentations_det = mask_augmentations.to_deterministic()

        mask_augmentations_det = mask_augmentations_det.copy_random_state(
            image_augmentations_det,
            matching='name'
        )

        batch_x = image_augmentations_det.augment_images(batch_x)
        batch_y = mask_augmentations_det.augment_images(batch_y)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_names)