#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/
# REPO_PATH="/home/hqs5468/hua/workspace/projects/convxai/src/convxai/writing_models"
# export PYTHONPATH=$REPO_PATH

CUDA_VISIBLE_DEVICES=1 python ../trainers/quality_trainer/main.py \
                    --config_dir "../configs/quality_model_config.json"  
