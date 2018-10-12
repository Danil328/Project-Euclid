import gc
import os
import pandas as pd
from keras import optimizers
from data import create_stratified_validation, get_augmentations, DataGenerator
from utils import read_train_data_to_memory, get_folds_iterations
from utils import take_image_names, callbacks_factory
from models import make_model
from losses import bce_jaccard_loss
from metrics import map_at_different_iou, hard_jaccard_coef

import matplotlib.pyplot as plt

FOLDS = 6
IS_TEST = True
STRATIFIED_BY='all'
RANDOM_STATE = 17
EPOCHS = 15

NETWORK = 'vanilla_unet'
WITH_SIGMOID = True
WITH_WEIGHTS = True
HEIGHT = 100
WIDTH = 100
INPUT_PADDING = 184
BATCH_SIZE = 12


def main():
    folds_df = create_stratified_validation(n_folds=FOLDS, stratified_by=STRATIFIED_BY)
    folds_iterations = get_folds_iterations(FOLDS - 1) if IS_TEST else get_folds_iterations(FOLDS)
    images_dict, masks_dict = read_train_data_to_memory()

    for train_folds, valid_fold in folds_iterations:
        train_generator = DataGenerator(
            images_dict,
            masks_dict,
            image_names=take_image_names(folds_df, train_folds),
            augmentations=get_augmentations(),
            batch_size=BATCH_SIZE,
            shape=(HEIGHT, WIDTH),
            input_padding=INPUT_PADDING
        )
        valid_generator = DataGenerator(
            images_dict,
            masks_dict,
            image_names=take_image_names(folds_df, valid_fold),
            batch_size=BATCH_SIZE,
            shape=(HEIGHT, WIDTH),
            input_padding=INPUT_PADDING
        )

        callbacks = callbacks_factory(
            callbacks_list=['best_model_checkpoint', 'early_stopping', 'tensorboard', 'csv_logger', 'learning_rate_scheduler'],
            model_maskname='{0}_{1}_fold_{2}'.format('standard', NETWORK, valid_fold[0]),
            monitor='val_map_at_different_iou'
        )

        height = HEIGHT + INPUT_PADDING if INPUT_PADDING else HEIGHT
        width = WIDTH + INPUT_PADDING if INPUT_PADDING else WIDTH
        model = make_model(
            network=NETWORK,
            input_shape=(height, width, 3),
            with_sigmoid=True,
            random_state=RANDOM_STATE
        )

        if WITH_WEIGHTS:
            weights_pathway = '../../data/intermediate/{0}_weights.h5'.format(NETWORK)
            print('Loading weights from {0}'.format(weights_pathway))
            model.load_weights(weights_pathway, by_name=True)
        else:
            print('No weights passed, training from scratch')

        model.compile(
            loss=bce_jaccard_loss,
            optimizer=optimizers.RMSprop(lr=0.001, decay=0.001),
            metrics=[hard_jaccard_coef, map_at_different_iou])

        model.fit_generator(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=valid_generator,
            verbose=1,
            workers=4,
            use_multiprocessing=False,
        )

        if IS_TEST:
            test_generator = DataGenerator(
                images_dict,
                masks_dict,
                image_names=take_image_names(folds_df, [FOLDS - 1]),
                batch_size=BATCH_SIZE,
                shape=(HEIGHT, WIDTH),
                input_padding=INPUT_PADDING
            )
            results = model.evaluate_generator(test_generator)
            results_pathway = '../../logs/test_evaluation_{0}_fold_{1}.csv'.format(NETWORK, valid_fold[0])
            pd.DataFrame({
                'MetricsNames': model.metrics_names,
                'Results': results
            }).to_csv(os.path.join(results_pathway), index=False)

        break
        # del model
        # K.clear_session()
        # gc.collect()


if __name__ == '__main__':
    main()