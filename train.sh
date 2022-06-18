#!/bin/bash

echo Enter the path to logs
# shellcheck disable=SC2162
read -p 'Path: ' logs
echo Enter the path to the best model
read -p 'Path: ' model

python train.py --work-dir $logs \
       --data-path datasets_parsed_motion_inpaint_64with_object/ \
       --inr-config config/inr-gan.yml \
       --batch-size 16 \
       --use-multigpu false \
       --n-workers 8 \
       --n-epochs 500 \
       --lr 5e-4 \
       --best-model $model