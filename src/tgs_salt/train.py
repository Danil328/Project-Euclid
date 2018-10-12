import gc
import os
import warnings
import pandas as pd
from keras import optimizers
from data import create_stratified_validation, get_augmentations, DataGenerator
from utils import read_train_data_to_memory, get_folds_iterations
from utils import take_image_names, callbacks_factory
from models import make_model
from losses import bce_jaccard_loss
from metrics import map_at_different_iou, hard_jaccard_coef
from params import args

warnings.filterwarnings('ignore')


def main():
    folds_df = create_stratified_validation(n_folds=args.folds, stratified_by=args.stratified_by)
    folds_iterations = get_folds_iterations(args.folds - 1) if args.is_test else get_folds_iterations(args.folds)
    images_dict, masks_dict = read_train_data_to_memory()

    for train_folds, valid_fold in folds_iterations:
        train_generator = DataGenerator(
            images_dict,
            masks_dict,
            image_names=take_image_names(folds_df, train_folds),
            augmentations=get_augmentations(),
            batch_size=args.batch_size,
            shape=(args.height, args.width),
            input_padding=args.input_padding
        )
        valid_generator = DataGenerator(
            images_dict,
            masks_dict,
            image_names=take_image_names(folds_df, valid_fold),
            batch_size=args.batch_size,
            shape=(args.height, args.width),
            input_padding=args.input_padding
        )

        callbacks = callbacks_factory(
            callbacks_list=['best_model_checkpoint', 'early_stopping', 'tensorboard', 'csv_logger', 'learning_rate_scheduler'],
            model_maskname='{0}_{1}_fold_{2}'.format('standard', args.network, valid_fold[0]),
            monitor='val_map_at_different_iou'
        )

        height = args.height + args.input_padding if args.input_padding else args.height
        width = args.width + args.input_padding if args.input_padding else args.width
        model = make_model(
            network=args.network,
            input_shape=(height, width, 3),
            with_sigmoid=True,
            random_state=args.random_state
        )

        if args.with_weights:
            weights_pathway = '../../data/intermediate/{0}_weights.h5'.format(args.network)
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
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=valid_generator,
            verbose=1,
            workers=4,
            use_multiprocessing=False,
        )

        if args.is_test:
            test_generator = DataGenerator(
                images_dict,
                masks_dict,
                image_names=take_image_names(folds_df, [args.folds - 1]),
                batch_size=args.batch_size,
                shape=(args.height, args.width),
                input_padding=args.input_padding
            )
            results = model.evaluate_generator(test_generator)
            results_pathway = '../../logs/test_evaluation_{0}_fold_{1}.csv'.format(args.network, valid_fold[0])
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