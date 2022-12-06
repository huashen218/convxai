#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/huashen218/workspace/convxai
RUN_SERVICE_DIR="/home/huashen218/workspace/convxai/convxai";

# export PYTHONPATH=/home/hqs5468/hua/workspace/projects/convxai/
# RUN_SERVICE_DIR="/home/hqs5468/hua/workspace/projects/convxai/convxai";

python $RUN_SERVICE_DIR//services/web_service/web_server.py
