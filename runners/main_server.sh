#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/huashen218/workspace/convxai_system/
RUN_SERVICE_DIR="/home/huashen218/workspace/convxai_system/convxai/services/run_server";

# CUDA_VISIBLE_DEVICES=0 
python $RUN_SERVICE_DIR/run.py \
                        --config-path $RUN_SERVICE_DIR/config.yml \
                        --port 10020;