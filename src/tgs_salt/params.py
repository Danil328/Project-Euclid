import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--network', default='vanilla_unet')
arg('--alias', default='')
arg('--test_mode', type=lambda x: (str(x).lower() == 'true'), default=True)
arg('--valid_fold', type=int, default=1)
arg('--n_folds', type=int, default=6)
arg('--stratified_by', default='all')
arg('--random_state', type=int, default=17)
arg('--weights', default='')
arg('--height', type=int, default=100)
arg('--width', type=int, default=100)
arg('--channels', type=int, default=3)
arg('--input_padding', type=int, default=184)
arg('--batch_size', type=int, default=12)
arg('--loss_function', default='focal_loss')
arg('--optimizer', default='rmsprop')
arg('--learning_rate', type=float, default=.001)
arg('--decay', type=float, default=.001)
arg('--epochs', type=int, default=100)
arg('--zca_whitening', type=lambda x: (str(x).lower() == 'true'), default=False)
arg('--zca_epsilon', type=float, default=10.)
arg('--cumsum', type=lambda x: (str(x).lower() == 'true'), default=False)

arg('--monitor', default='val_conv_u0d-score_map_at_different_iou')
arg('--monitor_mode', default='max')
arg('--early_stopping_patience', type=int, default=10)

arg('--source_dir', default='../../data/source')
arg('--train_dir', default='../../data/source/train')
arg('--test_dir', default='../../data/source/test')
arg('--intermediate_dir', default='../../data/intermediate')
arg('--models_dir', default='../../models')
arg('--logs_dir', default='../../logs')

args = parser.parse_args()