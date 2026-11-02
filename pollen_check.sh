#!/usr/bin/env bash
# Thử não Pollen trên MicroDuck Isaac Lab: ./pollen_check.sh walk|getup|kick [thêm tham số check_pollen.py]
here=$(cd "$(dirname "$0")" && pwd)
docker run --rm --device nvidia.com/gpu=all --shm-size 8g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -w /workspace/isaaclab \
  -v "$here/microduck_course":/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course:ro \
  --entrypoint /opt/isaaclab-venv/bin/python isaac-lab-main:3.0.0 \
  /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course/check_pollen.py --test "$@" 2>&1 \
  | grep -E "POLLEN|Traceback|Error|error" -A2
