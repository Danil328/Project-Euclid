import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--folds', type=int, default=6)
arg('--is_test', type=lambda x: (str(x).lower() == 'true'), default=True)
arg('--stratified_by', default='all')
arg('--random_state', type=int, default=777)
arg('--epochs', type=int, default=15)
arg('--network', default='vanilla_unet')
arg('--with_sigmoid', type=lambda x: (str(x).lower() == 'true'), default=True)
arg('--with_weights', type=lambda x: (str(x).lower() == 'true'), default=True)
arg('--height', type=int, default=100)
arg('--width', type=int, default=100)
arg('--input_padding', type=int, default=184)
arg('--batch_size', type=int, default=12)

args = parser.parse_args()