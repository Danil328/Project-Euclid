#!/usr/bin/env bash
cd ../src/tgs_salt/

python train.py \
--network vanilla_unet \
--test_mode True \
--valid_fold 1 \
--n_folds 6 \
--stratified_by all \
--random_state 17 \
--height 100 \
--width 100 \
--channels 3 \
--input_padding 184 \
--batch_size 12 \
--loss_function focal_dice_loss \
--optimizer rmsprop \
--learning_rate .001 \
--decay .001 \
--epochs 100 \
--zca_whitening False \
--zca_epsilon 10. \
--cumsum False \
--monitor val_conv_u0d-score_map_at_different_iou \
--monitor_mode max \
--early_stopping_patience 10