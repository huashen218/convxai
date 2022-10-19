#!/usr/bin/env bash
set -x;
set -e;
export PYTHONPATH=/data/hua/workspace/projects/convxai_system/

RUN_SERVICE_DIR="/data/hua/workspace/projects/convxai_system/convxai/services/web_service";
python $RUN_SERVICE_DIR/web_server.py \
        # --port 12211;