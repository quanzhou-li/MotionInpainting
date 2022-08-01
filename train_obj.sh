#!/bin/bash

echo Enter the path to logs
# shellcheck disable=SC2162
read -p 'Path: ' logs
echo Enter the path to the best model
read -p 'Path: ' model

python train_obj.py --work-dir $logs \
       --data-path datasets_parsed_motion_infill_startfromgrasp/ \
       --inr-config config/inr-gan.yml \
       --batch-size 10 \
       --use-multigpu false \
       --n-workers 8 \
       --n-epochs 1500 \
       --lr 5e-4 \
       # --best-model $model