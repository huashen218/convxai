run_stage_one.sh#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/

CUDA_VISIBLE_DEVICES=0 python run_stage_one.py \
                        -task diversity \
                        -stage1_exp diversity_model \
                        -model_max_length 512 \
                        -num_epochs 20 \
                        -train_batch_size 1 \
                        -grad_type "integrated_l1" \
                        -results_dir "/data/hua/workspace/projects/convxai/src/convxai/xai_models/checkpoints/xai_counterfactual_explainer_models/"