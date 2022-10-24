#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/


CUDA_VISIBLE_DEVICES=0 python run_stage_two.py \
                        -task diversity \
                        -stage2_exp diversity_counterfactual \
                        -editor_path /data/hua/workspace/projects/convxai/src/convxai/xai_models/checkpoints/xai_counterfactual_explainer_models/diversity/editors/diversity_model/checkpoints/best.pth



