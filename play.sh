#!/usr/bin/env bash
# Xem MicroDuck trên trình duyệt (Viser), hai sân chạy song song được:
#   ./play.sh course [checkpoint|latest]   sân "bàn làm việc tí hon", 9 con vịt  → http://<máy>:8090/
#   ./play.sh soccer [checkpoint|latest]   sân bóng, bộ chiến thuật viết tay       → http://<máy>:8091/
#   ./play.sh score  [checkpoint|latest]   4 sân, vịt dùng bộ não đá bóng đã học (RL) → http://<máy>:8092/
# course/soccer dùng chính sách đi bộ MicroDuck ("latest" = checkpoint mới nhất của lượt đất phẳng hoặc địa hình);
# score dùng chính sách đá bóng ("latest" = checkpoint mới nhất của microduck_soccer_score).
# Chạy song song được với train; chạy lại để nạp checkpoint mới hơn. Log ghi bàn: docker logs isaac-lab-soccer | grep SOCCER
# SOCCER_TRACE=1 ./play.sh soccer: in vị trí vịt, bóng, lệnh mỗi ~2 s và lý do kết thúc mỗi lượt.
mode=${1:-course}; ckpt=${2:-latest}
case $mode in
  course) task=IsaacContrib-Velocity-Course-MicroDuck; n=9; port=8090; name=isaac-lab-play ;;
  soccer) task=IsaacContrib-Soccer-MicroDuck; n=1; port=8091; name=isaac-lab-soccer ;;
  score) task=IsaacContrib-Soccer-Score-MicroDuck; n=4; port=8092; name=isaac-lab-score ;;
  *) echo "Chế độ không rõ: $mode (course|soccer)" >&2; exit 1 ;;
esac
here=$(cd "$(dirname "$0")" && pwd)
logs="$HOME/Documents/Codex/isaaclab/logs"
if [ "$ckpt" = latest ] && [ "$mode" != score ]; then
  # "--checkpoint latest" của Isaac Lab chỉ nhận run cùng tên task, nên tự tìm file mới nhất của các run
  # đất phẳng và địa hình (cùng 61 đầu vào cho actor nên dùng được cả hai).
  ckpt=$(ls -t "$logs"/rsl_rl/microduck_velocity_{flat,rough}/*/model_*.pt | head -1)
  ckpt=/workspace/isaaclab/logs/${ckpt#"$logs"/}
fi
echo "$task, checkpoint: $ckpt"
docker rm -f $name >/dev/null 2>&1 || true
docker run -d --name $name -e SOCCER_TRACE=${SOCCER_TRACE:-} --device nvidia.com/gpu=all --shm-size 4g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -w /workspace/isaaclab -p $port:8080 \
  -v "$logs":/workspace/isaaclab/logs \
  -v "$here/microduck_course":/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_course:ro \
  -v "$HOME/Documents/Codex/simready":/simready:ro -v "$HOME/Documents/Codex/simready_static":/simready_static:ro \
  --entrypoint /opt/isaaclab-venv/bin/isaaclab isaac-lab-main:3.0.0 \
  play --rl_library rsl_rl --task "$task" --checkpoint "$ckpt" --num_envs "$n" --viz viser
