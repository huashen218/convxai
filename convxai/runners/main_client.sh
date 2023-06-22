#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/huashen218/workspace/convxai
RUN_SERVICE_DIR="/home/huashen218/workspace/convxai/convxai";

python $RUN_SERVICE_DIR/services/web_service/web_server.py
