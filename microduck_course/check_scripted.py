"""Đo đội chiến thuật viết tay (mappo.scripted_team) đá với chính nó: người lấy bóng có tới được bóng, quay đúng
hướng, sút không. In phân bố vị trí bóng trong khung thân của người lấy bóng và số bàn / sút.

  python check_scripted.py --num_envs 32 --seconds 60
"""

import argparse
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seconds", type=float, default=60.0)
add_launcher_args(parser)
parser.set_defaults(device=None)
args, _ = parser.parse_known_args()
sys.argv = sys.argv[:1]
os.environ["SOCCER_TEAM_SIZE"] = "2"
args.visualizer = ["none"]

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from isaaclab_tasks.contrib.microduck_course.mappo import scripted_team  # noqa: E402
from isaaclab_tasks.contrib.microduck_course.soccer_team import TASK, TeamVecEnv  # noqa: E402

env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = args.num_envs
args.device = env_cfg.sim.device

with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    vec = TeamVecEnv(env, 2)
    n, m, a_dim = vec.n, vec.matches, vec.num_actions
    obs = vec.get_observations()
    near = far = kick_cmd = 0
    xs, ys, yaw_err = [], [], []
    totals0 = vec.totals.clone()
    for _ in range(int(args.seconds / env.step_dt)):
        o = obs["policy"].view(m, 2 * n, -1)
        act = torch.cat((scripted_team(o[:, :n], n, a_dim), scripted_team(o[:, n:], n, a_dim)), 1)
        flat = o.reshape(-1, o.shape[-1])
        ball = flat[:, 6:8]
        d = torch.linalg.norm(ball, dim=-1)
        goal = flat[:, 10:12]
        close = d < 0.25
        near += int(close.sum())
        far += int((~close).sum())
        kick_cmd += int((act.reshape(-1, a_dim)[:, 3] > 0.5).sum())
        if close.any():
            xs.append(ball[close, 0])
            ys.append(ball[close, 1])
            to_goal = goal[close] - ball[close]
            yaw_err.append(torch.atan2(to_goal[:, 1], to_goal[:, 0]).abs())
        obs, *_ = vec.step(act.reshape(-1, a_dim))
    c = dict(zip(TeamVecEnv.STAT_NAMES, (vec.totals - totals0).tolist()))
    minutes = args.seconds / 60.0 * m
    x, y, e = torch.cat(xs), torch.cat(ys), torch.cat(yaw_err)
    q = lambda t: [round(float(v), 3) for v in torch.quantile(t, torch.tensor([0.1, 0.5, 0.9], device=t.device))]  # noqa: E731
    print(f"[SCRIPTED] {m} trận x {args.seconds:.0f} s: bàn {float(vec.goals.sum()):.0f} ({float(vec.goals.sum()) / minutes:.2f}/phút), "
          f"sút {c['kick'] / minutes:.2f}/phút, trúng khung {c['shot'] / minutes:.2f}/phút, ném vào {c['throw_in'] / minutes:.2f}/phút, "
          f"ngã {c['fall'] / minutes:.2f}/phút, bóng đứng yên {c['stalled'] / max(c['steps'], 1):.0%}", flush=True)
    print(f"[SCRIPTED] vịt ở trong 0,25 m quanh bóng {near / (near + far):.1%} thời gian; lệnh sút {kick_cmd} lần", flush=True)
    print(f"[SCRIPTED] khi gần bóng: bóng x thân (10/50/90%) {q(x)}, y {q(y)}, lệch hướng khung thành (rad) {q(e)}", flush=True)
    env.close()
