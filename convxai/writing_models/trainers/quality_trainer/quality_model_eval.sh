#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/

CUDA_VISIBLE_DEVICES=1 python ../trainers/quality_trainer/ppl_distribution.py \
                    --config_dir "../configs/quality_model_config.json" \
                    --save_ppl_dir "../checkpoints/quality_model/"