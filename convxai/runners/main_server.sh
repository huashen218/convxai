#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/huashen218/workspace/convxai/
RUN_SERVICE_DIR="/home/huashen218/workspace/convxai/convxai";

# CUDA_VISIBLE_DEVICES=0 
python $RUN_SERVICE_DIR/services/run_server/run.py \
                        --config-path $RUN_SERVICE_DIR/configs/service_config.yml \
                        --port 10020;