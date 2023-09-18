#!/bin/sh

MODEL_FLAGS_1="--batch-size 64 --save-dir train_pts/batch_64_centroid_25_5fold_crossval --centroid-weight 25"
MODEL_FLAGS_2="--batch-size 64 --save-dir train_pts/batch_64_centroid_30_5fold_crossval --centroid-weight 30"
MODEL_FLAGS_3="--batch-size 64 --save-dir train_pts/batch_64_centroid_35_5fold_crossval --centroid-weight 35"
MODEL_FLAGS_4="--batch-size 64 --save-dir train_pts/batch_64_centroid_40_5fold_crossval --centroid-weight 40"

python loc_train.py $MODEL_FLAGS_1
python loc_train.py $MODEL_FLAGS_2
python loc_train.py $MODEL_FLAGS_3
python loc_train.py $MODEL_FLAGS_4