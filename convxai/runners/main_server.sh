#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/hqs5468/hua/workspace/projects/convxai/
RUN_SERVICE_DIR="/home/hqs5468/hua/workspace/projects/convxai/convxai";

CUDA_VISIBLE_DEVICES=2 python $RUN_SERVICE_DIR/services/run_server/run.py \
                        --config-path $RUN_SERVICE_DIR/configs/service_config.yml \
                        --port 10020;