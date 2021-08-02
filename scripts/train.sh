#!/usr/bin/env bash

set -x

CONFIG=$1

# python -u scripts/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher pytorch


# TODO:
# 1. wrong in readme? `--use_KL True` seems L_vlb
# 2. what is rescale loss term?
# 3. how to calculate cdf
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \
    scripts/train.py $CONFIG --launcher pytorch ${@:3}
