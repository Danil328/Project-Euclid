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
            zca_whitening=None,
            augmentations=None,
            batch_size=32,
            input_shape=(101, 101),
            input_padding=0,
            dataset_shape=(101, 101),
            channels=1,
            shuffle=True):

        self.images_dict = images_dict
        self.masks_dict = masks_dict
        self.image_names = sorted(image_names)
        self.zca_whitening=zca_whitening
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.images_input_shape = (input_shape[0] + input_padding, input_shape[1] + input_padding)
        self.masks_input_shape = input_shape
        self.dataset_shape = dataset_shape
        self.channels=channels
        self.shuffle = shuffle

        self.dim_pad = int(input_padding / 2)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_names) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image_names = self.image_names[index * self.batch_size: (index + 1) * self.batch_size]

        batch_x = np.empty((self.batch_size, *self.dataset_shape, self.channels), dtype=np.uint8)
        batch_y = np.empty((self.batch_size, *self.dataset_shape, 1), dtype=np.uint8)

        for i, image_name in enumerate(batch_image_names):
            batch_x[i, ...] = self.images_dict[image_name]
            batch_y[i, :, :, 0] = binarize(self.masks_dict[image_name][..., 0], 256 / 2)

        if self.augmentations:
            batch_x, batch_y = self._apply_augmentations(batch_x, batch_y)

        if self.zca_whitening:
            batch_x = self._apply_zca_whitening(batch_x)

        batch_x_resized = np.empty((self.batch_size, *self.images_input_shape, self.channels), dtype=np.uint8)
        batch_aux3_resized = np.empty((self.batch_size, 32, 32, 1), dtype=np.uint8)
        batch_aux2_resized = np.empty((self.batch_size, 56, 56, 1), dtype=np.uint8)
        batch_aux1_resized = np.empty((self.batch_size, 104, 104, 1), dtype=np.uint8)
        batch_y_resized = np.empty((self.batch_size, *self.masks_input_shape, 1), dtype=np.uint8)

        for i in range(batch_x.shape[0]):
            image = batch_x[i]

            if self.channels == 3:
                image = imresize(image, self.masks_input_shape, interp='nearest')
                image = pad(image, ((self.dim_pad, self.dim_pad), (self.dim_pad, self.dim_pad), (0, 0)), mode='reflect')
                batch_x_resized[i, ...] = image

            else:
                image = image[..., 0]
                image = imresize(image, self.masks_input_shape, interp='nearest')
                image = pad(image, ((self.dim_pad, self.dim_pad), (self.dim_pad, self.dim_pad)), mode='reflect')
                batch_x_resized[i, :, :, 0] = image

            mask = batch_y[i][..., 0]
            batch_aux3_resized[i, :, :, 0] = imresize(mask, (32, 32), interp='nearest')
            batch_aux2_resized[i, :, :, 0] = imresize(mask, (56, 56), interp='nearest')
            batch_aux1_resized[i, :, :, 0] = imresize(mask, (104, 104), interp='nearest')
            batch_y_resized[i, :, :, 0] = imresize(mask, self.masks_input_shape, interp='nearest')

        return batch_x_resized / 255., {'conv_aux3_score': batch_aux3_resized,
                                        'conv_aux2_score': batch_aux2_resized,
                                        'conv_aux1_score': batch_aux1_resized,
                                        'conv_u0d-score': batch_y_resized}

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

    def _apply_zca_whitening(self, batch_x):
        image_vectors = batch_x.reshape((-1, batch_x.shape[1] * batch_x.shape[2] * batch_x.shape[3]))
        zca_images_vectors = self.zca_whitening.transform(image_vectors)

        return zca_images_vectors.reshape(batch_x.shape)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_names)