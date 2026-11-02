#!/usr/bin/env bash
# Kiểm tra trọng tài + kỹ năng của trận đồng đội (microduck_course/check_scripted.py).
here=$(cd "$(dirname "$0")" && pwd)
docker run --rm --device nvidia.com/gpu=all --shm-size 8g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -w /workspace/isaaclab \
  -v "$here/microduck_course":/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course:ro \
  --entrypoint /opt/isaaclab-venv/bin/python isaac-lab-main:3.0.0 \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course/check_scripted.py "$@" 2>&1 \
  | grep -E "SCRIPTED|Traceback|Error|error" -A4
