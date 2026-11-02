"""Kiểm tra trọng tài và kỹ năng của TeamVecEnv (2 đấu 2), mỗi trận một tình huống:

  0. bóng nằm ngoài biên dọc → người vô hình ném bóng bay vào sân (không dịch chuyển bóng), bóng rơi trong sân;
  1. bóng vào lưới đội B → bàn cho A, bóng về giữa sân, các vịt tự chạy về vị trí, rồi đội B giao bóng;
  2. lật ngửa vịt 0 → vịt chuyển sang kỹ năng đứng dậy và tự đứng dậy, không bị dựng lại;
  3. đặt bóng trước chân vịt 1 và ra lệnh sút → kỹ năng sút chạy, bóng bay đi.
  python check_rules.py   (trong container, như team.py)
"""

import argparse
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
parser.set_defaults(device=None)
args, _ = parser.parse_known_args()
sys.argv = sys.argv[:1]
os.environ["SOCCER_TEAM_SIZE"] = "2"
args.visualizer = ["none"]

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from isaaclab_tasks.contrib.microduck_course.soccer_rl import HighLevelVelocityAction as Skill  # noqa: E402
from isaaclab_tasks.contrib.microduck_course.soccer_team import BALL_R, TASK, TeamVecEnv  # noqa: E402

env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 4
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    vec = TeamVecEnv(env, 2)
    hx, hy = vec.half_x, vec.half_y
    dev, dt = env.device, env.step_dt
    origins = env.scene.env_origins
    ok = True

    def report(name, good, detail):
        global ok
        ok &= bool(good)
        print(f"[RULES] {name}: {detail}{'' if good else '  <-- SAI'}", flush=True)

    def ball_local(i):
        return vec.ball.data.root_pos_w.torch[i] - origins[i]

    zero = torch.zeros(vec.matches, vec.k, vec.num_actions, device=dev)
    zero[..., 3] = -1.0  # không sút
    # Đứng yên 1 s cho vịt ổn định.
    for _ in range(int(1.0 / dt)):
        vec.step(zero.view(-1, vec.num_actions))
    vec.restart_team[:] = -1  # bỏ bảo vệ giao bóng đầu trận

    # 0: bóng ngoài biên dọc.  1: bóng trong lưới B.  3: bóng trước chân phải vịt 1.
    pose = torch.zeros(vec.matches, 7, device=dev)
    pose[:, 6] = 1.0
    pose[0, :3] = origins[0] + torch.tensor((0.3, hy + 0.2, BALL_R + 0.005), device=dev)
    pose[1, :3] = origins[1] + torch.tensor((hx + 0.1, 0.0, BALL_R + 0.005), device=dev)
    duck = vec.robots[1]
    yaw = float(duck.data.heading_w.torch[3])
    off = (0.09, -0.042)  # bóng trước chân phải, trong khung hướng của vịt
    p = duck.data.root_pos_w.torch[3]
    pose[3, 0] = p[0] + math.cos(yaw) * off[0] - math.sin(yaw) * off[1]
    pose[3, 1] = p[1] + math.sin(yaw) * off[0] + math.cos(yaw) * off[1]
    pose[3, 2] = origins[3, 2] + BALL_R + 0.001
    keep = torch.tensor([0, 1, 3], device=dev)
    vec.ball.write_root_pose_to_sim_index(root_pose=pose[keep], env_ids=keep)
    vec.ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(3, 6, device=dev), env_ids=keep)
    # 2: lật ngửa vịt 0.
    r0 = vec.robots[0]
    rp = r0.data.root_link_pose_w.torch[2:3].clone()
    rp[0, 2] = origins[2, 2] + 0.07
    rp[0, 3:] = torch.tensor((0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)), device=dev)
    r0.write_root_pose_to_sim_index(root_pose=rp, env_ids=torch.tensor([2], device=dev))

    act = zero.clone()
    act[3, 1, 3] = 1.0  # trận 3, vịt 1: sút
    start0 = ball_local(0)[:2].clone()
    goals0 = vec.match_score.clone()
    flight_seen, kick_seen, kick_speed, down_seen, up_at = False, False, 0.0, False, None
    return_started, return_done_at = False, None
    totals0 = vec.totals.clone()
    for k in range(int(8.0 / dt)):
        vec.step((act if k < 2 else zero).view(-1, vec.num_actions))
        t = (k + 1) * dt
        flight_seen |= bool(vec.in_flight[0])
        b3 = vec.ball.data.root_lin_vel_w.torch[3, :2]
        kick_seen |= int(vec.terms[1].mode[3]) == Skill.KICK
        kick_speed = max(kick_speed, float(torch.linalg.norm(b3)))
        m0 = int(vec.terms[0].mode[2])
        down_seen |= m0 == Skill.DOWN
        if down_seen and up_at is None and m0 == Skill.WALK:
            up_at = t
        return_started |= bool(vec.returning[1])
        if return_started and return_done_at is None and not bool(vec.returning[1]):
            return_done_at = t
    counts = dict(zip(TeamVecEnv.STAT_NAMES, (vec.totals - totals0).tolist()))
    print(f"[RULES] đếm sự kiện: {counts}", flush=True)

    b0 = ball_local(0)
    report("0 ném bóng vào sân", flight_seen and abs(float(b0[0])) < hx and abs(float(b0[1])) < hy,
           f"bóng bay {flight_seen}, ra ở ({start0[0]:+.2f}, {start0[1]:+.2f}) → giờ ở ({b0[0]:+.2f}, {b0[1]:+.2f})")
    g = vec.match_score[1] - goals0[1]
    report("1 bàn thắng + chạy về", int(g[0]) == 1 and return_started and return_done_at is not None,
           f"bàn A +{int(g[0])} B +{int(g[1])}, chạy về xong lúc {return_done_at} s, "
           f"dịch chuyển dự phòng {int(counts['return_teleport'])} vịt, giao bóng đội {int(vec.kickoff_team[1])} (cần 1)")
    report("2 ngã tự đứng dậy", down_seen and up_at is not None and counts["unrecovered"] == 0,
           f"chuyển sang đứng dậy {down_seen}, đứng vững lúc {up_at} s, phải dựng lại {int(counts['unrecovered'])}")
    report("3 sút", kick_seen and kick_speed > 0.5, f"kỹ năng sút chạy {kick_seen}, tốc độ bóng đỉnh {kick_speed:.2f} m/s")
    print("[RULES] ĐÚNG HẾT" if ok else "[RULES] CÓ SAI", flush=True)
    env.close()
