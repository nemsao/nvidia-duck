"""Kiểm tra chính sách đi bộ đã xuất (policies/walk_rough.pt).

  --task IsaacContrib-Velocity-Flat-MicroDuck      : chạy file .pt trên bài đi bộ gốc (lệnh ngẫu nhiên của bài).
  --task IsaacContrib-Soccer-Score-MicroDuck       : chạy bài đá bóng với hành động tầng trên = 0 (đứng yên).
In tỷ lệ lượt bị ngã và độ cao thân sau N bước.
"""

import argparse
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=100)
add_launcher_args(parser)
parser.set_defaults(device=None, visualizer=["none"])
args, hydra_args = setup_preset_cli(parser, None)
sys.argv = [sys.argv[0]] + hydra_args

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
args.device = env_cfg.sim.device
with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    policy = torch.jit.load(os.path.join(os.path.dirname(__file__), "policies", "walk_rough.pt")).to(u.device).eval()
    obs, _ = env.reset()
    falls = torch.zeros(u.num_envs, device=u.device)
    for i in range(args.steps):
        with torch.inference_mode():
            if "Soccer" in args.task:
                act = torch.zeros((u.num_envs, 3), device=u.device)
            else:
                act = policy(obs["policy"])
            obs, _, term, _, _ = env.step(act)
        falls += term.float()
        if i % 20 == 0:
            z = u.scene["robot"].data.root_pos_w.torch[:, 2]
            print(f"[CHECK] bước {i}: độ cao thân trung bình {float(z.mean()):.3f} m, đã ngã {int(falls.sum())} lần", flush=True)
    obs_dim = obs["policy"].shape[-1]
    print(f"[CHECK] {args.task}: {args.steps} bước, {u.num_envs} vịt, ngã {int(falls.sum())} lần; obs policy {obs_dim} chiều", flush=True)
    env.close()
