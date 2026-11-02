#!/usr/bin/env bash
# Trận MicroDuck N đấu N (gói microduck_course/soccer_team.py), chạy trong container.
#   ./team.sh train 2 [--num_envs 512] [--max_iterations 2000] [--resume <đường dẫn trong logs/>]   (chạy nền)
#   ./team.sh play  2 <checkpoint trong logs/>                                                       → http://<máy>:8093/
#   ./team.sh eval  2 <checkpoint trong logs/> [--num_envs 64] [--seconds 300]                       (in số đo, chờ xong)
# Theo dõi train: docker logs -f isaac-lab-team-train | grep -E "Learning iteration|Mean reward|goals"
mode=${1:?train|play|eval}; n=${2:?số vịt mỗi đội}; shift 2
here=$(cd "$(dirname "$0")" && pwd)
logs="$HOME/Documents/Codex/isaaclab/logs"
mkdir -p "$logs" && chmod 777 "$logs"
script=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course/team.py
common=(--device nvidia.com/gpu=all --shm-size 8g -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -w /workspace/isaaclab
  -v "$logs":/workspace/isaaclab/logs
  -v "$here/microduck_course":/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course:ro
  --entrypoint /opt/isaaclab-venv/bin/python)
if [ "$mode" = train ]; then
  docker rm -f isaac-lab-team-train >/dev/null 2>&1 || true
  docker run -d --name isaac-lab-team-train "${common[@]}" isaac-lab-main:3.0.0 $script train --team_size "$n" "$@"
elif [ "$mode" = eval ]; then
  ckpt=${1:?Thiếu checkpoint}; shift
  docker run --rm --name isaac-lab-team-eval "${common[@]}" isaac-lab-main:3.0.0 $script eval \
    --team_size "$n" --checkpoint "/workspace/isaaclab/logs/$ckpt" "$@" 2>&1 | grep -E "\[EVAL\]|Error|error"
else
  ckpt=${1:?Thiếu checkpoint}; shift
  docker rm -f isaac-lab-team-play >/dev/null 2>&1 || true
  docker run -d --name isaac-lab-team-play -p 8093:8080 "${common[@]}" isaac-lab-main:3.0.0 $script play \
    --team_size "$n" --checkpoint "/workspace/isaaclab/logs/$ckpt" --num_envs 1 --viz viser "$@"
fi
