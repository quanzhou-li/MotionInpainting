#!/bin/bash

python train.py --work-dir logs \
       --data-path datasets_parsed_motion_imgs/ \
       --inr-config config/inr-gan.yml \
       --batch-size 16 \
       --use-multigpu true \
       --lr 1e-3