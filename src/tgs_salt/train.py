import os
import warnings
import pandas as pd
from data import create_stratified_validation, get_train_folds, get_augmentations, dict_to_vectors
from data import DataGenerator, ZCAWhitening
from helpers import read_train_data_to_memory, take_image_names
from models import make_model
from losses import make_loss
from metrics import map_at_different_iou, hard_jaccard_coef
from train_utils import callbacks_factory, make_optimizer
from params import args

warnings.filterwarnings('ignore')


def main():
    print('[INFO] Creating validation...')
    folds_df = create_stratified_validation(args.n_folds, args.source_dir, args.stratified_by)
    train_folds = get_train_folds(args.n_folds, args.test_mode, args.valid_fold)

    print('[INFO] Loading data...')
    images_dict, masks_dict = read_train_data_to_memory(args.train_dir, args.channels, cumsum=args.cumsum)

    zca_whitening = None
    if args.zca_whitening:
        print('[INFO] Fitting zca whitening...')
        zca_whitening = ZCAWhitening(zca_epsilon=args.zca_epsilon)
        zca_whitening.fit(dict_to_vectors(images_dict))

    train_generator = DataGenerator(
        images_dict,
        masks_dict,
        image_names=take_image_names(folds_df, train_folds),
        zca_whitening=zca_whitening,
        augmentations=get_augmentations(),
        batch_size=args.batch_size,
        input_shape=(args.height, args.width),
        channels=args.channels,
        input_padding=args.input_padding
    )
    valid_generator = DataGenerator(
        images_dict,
        masks_dict,
        image_names=take_image_names(folds_df, [args.valid_fold]),
        zca_whitening=zca_whitening,
        batch_size=args.batch_size,
        input_shape=(args.height, args.width),
        channels=args.channels,
        input_padding=args.input_padding
    )

    callbacks = callbacks_factory(
        callbacks_list=[
            'best_model_checkpoint',
            'last_model_checkpoint',
            'early_stopping',
            'tensorboard',
            'csv_logger',
            'learning_rate_scheduler'
        ],
        model_maskname='{0}{1}_fold_{2}'.format(args.alias, args.network, args.valid_fold),
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        early_stopping_parience=args.early_stopping_patience,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir
    )

    print('[INFO] Creating model...')
    height = args.height + args.input_padding if args.input_padding else args.height
    width = args.width + args.input_padding if args.input_padding else args.width
    model = make_model(
        network=args.network,
        input_shape=(height, width, args.channels),
        random_state=args.random_state
    )

    if args.weights:
        print('[INFO] Loading weights from {0}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)
    else:
        print('[INFO] No weights passed, training from scratch')

    model.compile(
        loss={
            'conv_aux3_score': make_loss(args.loss_function),
            'conv_aux2_score': make_loss(args.loss_function),
            'conv_aux1_score': make_loss(args.loss_function),
            'conv_u0d-score': make_loss(args.loss_function)
        },
        loss_weights={
            'conv_aux3_score': 0.1,
            'conv_aux2_score': 0.2,
            'conv_aux1_score': 0.3,
            'conv_u0d-score': 0.4,
        },
        optimizer=make_optimizer(args.optimizer, args.learning_rate, args.decay),
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

    if args.test_mode:
        model.load_weights(os.path.join(args.models_dir, 'best_{0}{1}_fold_{2}.h5') \
                           .format(args.alias, args.network, args.valid_fold))
        test_generator = DataGenerator(
            images_dict,
            masks_dict,
            image_names=take_image_names(folds_df, [args.n_folds]),
            zca_whitening=zca_whitening,
            batch_size=args.batch_size,
            input_shape=(args.height, args.width),
            channels=args.channels,
            input_padding=args.input_padding
        )
        results = model.evaluate_generator(test_generator)
        results_pathway = os.path.join(args.logs_dir, 'test_{0}{1}_fold_{2}.csv').format(
            args.alias,
            args.network,
            args.valid_fold
        )
        pd.DataFrame({
            'MetricsNames': model.metrics_names,
            'Results': results
        }).to_csv(os.path.join(results_pathway), index=False)

    a = 4

if __name__ == '__main__':
    main()

    # import matplotlib.pyplot as plt
    #
    # batch = 3
    # n = 1
    #
    # batch_x, batch_y = train_generator[batch]
    #
    # plt.imshow(batch_x[n, :, :, 0][92:-92, 92:-92], cmap='gray')
    # plt.show()
    # plt.imshow(batch_y[n, :, :, 0])
    # plt.show()