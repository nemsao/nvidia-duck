#!/usr/bin/env bash
# Train các bài trong gói microduck_course (container isaac-lab-soccer-train, chạy nền, log ở isaaclab/logs/rsl_rl/).
#   ./train.sh IsaacContrib-Soccer-Score-MicroDuck [--num_envs 4096] [--max_iterations 1500] ...
# Theo dõi: docker logs -f isaac-lab-soccer-train | grep -E "Learning iteration|Mean reward|Episode_Termination"
task=${1:?Thiếu tên task}; shift
here=$(cd "$(dirname "$0")" && pwd)
logs="$HOME/Documents/Codex/isaaclab/logs"
docker rm -f isaac-lab-soccer-train >/dev/null 2>&1 || true
docker run -d --name isaac-lab-soccer-train --device nvidia.com/gpu=all --shm-size 8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -w /workspace/isaaclab \
  -v "$logs":/workspace/isaaclab/logs \
  -v "$here/microduck_course":/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course:ro \
  -v "$HOME/Documents/Codex/simready":/simready:ro -v "$HOME/Documents/Codex/simready_static":/simready_static:ro \
  --entrypoint /opt/isaaclab-venv/bin/isaaclab isaac-lab-main:3.0.0 \
  train --rl_library rsl_rl --task "$task" "$@"
