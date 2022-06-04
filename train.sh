#!/bin/bash

echo Enter the path to logs
# shellcheck disable=SC2162
read -p 'Path: ' logs

python train.py --work-dir $logs \
       --data-path datasets_parsed_motion_imgs/ \
       --inr-config config/inr-gan.yml \
       --batch-size 16 \
       --use-multigpu true \
       --n-workers 12 \
       --n-epochs 500 \
       --lr 5e-4