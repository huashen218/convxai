#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/

CUDA_VISIBLE_DEVICES=0 python ./main.py \
                    --config_dir "../configs/diversity_model_config.json"




