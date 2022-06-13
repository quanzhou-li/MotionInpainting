#!/bin/bash

echo Enter the path to logs
# shellcheck disable=SC2162
read -p 'Path: ' logs

python train.py --work-dir $logs \
       --data-path datasets_parsed_motion_inpaint_64with_object/ \
       --inr-config config/inr-gan.yml \
       --batch-size 16 \
       --use-multigpu false \
       --n-workers 1 \
       --n-epochs 300 \
       --lr 5e-4