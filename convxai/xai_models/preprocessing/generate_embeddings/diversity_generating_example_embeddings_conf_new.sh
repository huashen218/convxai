#!/usr/bin/env bash
set -x;
set -e;
REPO_PATH="/data/hua/workspace/projects/convxai/src/"
export PYTHONPATH=$REPO_PATH

CUDA_VISIBLE_DEVICES=1 python ./diversity_generating_example_embeddings_conf_new.py \
                    --config_dir "$REPO_PATH/convxai/writing_models/configs/diversity_model_config.json" \
                    --data_save_dir "$REPO_PATH/convxai/xai_models/checkpoints/xai_example_embeddings"


