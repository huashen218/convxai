#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/home/huashen218/workspace/convxai/
RUN_SERVICE_DIR="/home/huashen218/workspace/convxai/convxai/services/web_service";
python $RUN_SERVICE_DIR/web_server.py