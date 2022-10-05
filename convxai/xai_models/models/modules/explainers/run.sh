#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai/src/

CUDA_VISIBLE_DEVICES=0 python attribution_explainer.py