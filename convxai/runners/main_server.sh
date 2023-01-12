#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/data/hua/workspace/projects/convxai/
RUN_SERVICE_DIR="/data/data/hua/workspace/projects/convxai/convxai";

CUDA_VISIBLE_DEVICES=2 python $RUN_SERVICE_DIR/services/run_server/run.py \
                        --config-path $RUN_SERVICE_DIR/configs/service_config.yml \
                        --port 10020;