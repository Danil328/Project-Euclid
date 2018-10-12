#!/usr/bin/env bash
cd ../src/tgs_salt/

python train.py #\
#--folds 6 \
#--is_test True \
#--stratified_by 'all' \
#--random_state 17 \
#--epochs 15 \
#--network vanilla_unet \
#--with_sigmoid True \
#--with_weights True \
#--height 100 \
#--width 100 \
#--input_padding 184 \
#--batch_size 12