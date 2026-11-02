"""Thử não Pollen (policies/velstand.pt, ball_kick_*.pt; đổi bằng pollen_onnx.py) trên MicroDuck của Isaac Lab.

  --test walk   : lệnh vận tốc ngẫu nhiên của bài đi bộ; so velstand.pt với walk_rough.pt (số lần ngã, sai lệch vận tốc).
  --test getup  : vịt nằm ở 5 tư thế ngã (sấp, ngửa, nghiêng trái / phải, ngẫu nhiên), lệnh 0; velstand.pt phải tự
                  đứng dậy. In tỷ lệ đứng dậy và thời gian.
  --test kick   : vịt đứng, bóng 70 mm / 15 g ngay trước chân sút; chạy ball_kick_<foot>.pt 0,5 s rồi velstand.pt.
                  In tốc độ, hướng bóng, quãng đường và số vịt ngã.
Dùng thân va chạm toàn phần (MICRODUCK_ALLCOLLISIONS_CFG) cho getup / kick như bài train của Pollen.
"""

import argparse
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "IsaacContrib-Velocity-Flat-MicroDuck"
CMD = slice(48, 61)  # lệnh vận tốc, tư thế đầu, tư thế thân trong 61 chiều quan sát

parser = argparse.ArgumentParser()
parser.add_argument("--test", choices=["walk", "getup", "kick"], required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--foot", choices=["left", "right"], default="right")
parser.add_argument("--kick_s", type=float, default=0.5)
parser.add_argument("--allcoll", action="store_true", help="walk: dùng thân va chạm toàn phần")
add_launcher_args(parser)
parser.set_defaults(device=None, visualizer=["none"])
args, hydra_args = setup_preset_cli(parser, None)
sys.argv = [sys.argv[0]] + hydra_args

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_CFG  # noqa: E402

POLICIES = os.path.join(os.path.dirname(__file__), "policies")

env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = args.num_envs
args.device = env_cfg.sim.device
if args.test != "walk" or args.allcoll:
    env_cfg.scene.robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path=env_cfg.scene.robot.prim_path)
    env_cfg.sim.physics.solver_cfg.nconmax = 96
    env_cfg.sim.physics.solver_cfg.njmax = 768
if args.test != "walk":
    env_cfg.terminations.fell_over = None
    env_cfg.sim.physics.solver_cfg.nconmax = 96
    env_cfg.sim.physics.solver_cfg.njmax = 768
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 1e4
if args.test == "kick":
    env_cfg.scene.ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.04)),
    )


def quat_xyzw(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def tilt_deg(robot):
    return torch.rad2deg(torch.acos((-robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)))


with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    dev, n, dt = u.device, u.num_envs, u.step_dt
    robot = u.scene["robot"]
    load = lambda name: torch.jit.load(os.path.join(POLICIES, name)).to(dev).eval()  # noqa: E731
    velstand = load("velstand.pt")
    obs, _ = env.reset()

    if args.test == "walk":
        for name in ("walk_rough.pt", "velstand.pt"):
            policy = load(name)
            obs, _ = env.reset()
            falls, err, count = 0, 0.0, 0
            for _ in range(int(args.seconds / dt)):
                with torch.no_grad():
                    obs, _, term, _, _ = env.step(policy(obs["policy"]))
                cmd = u.command_manager.get_command("base_velocity")
                vel = torch.cat((robot.data.root_lin_vel_b.torch[:, :2], robot.data.root_ang_vel_b.torch[:, 2:3]), -1)
                alive = ~term
                err += float((cmd - vel).abs()[alive].sum(0).sum())
                count += int(alive.sum())
                falls += int(term.sum())
            print(f"[POLLEN] đi bộ {name}: {n} vịt x {args.seconds:.0f} s, ngã {falls} lần, "
                  f"sai lệch |lệnh - vận tốc| trung bình (vx+vy+wz) {err / max(count, 1):.3f}", flush=True)

    elif args.test == "getup":
        poses = [("nằm sấp", (0.0, math.pi / 2)), ("nằm ngửa", (0.0, -math.pi / 2)),
                 ("nghiêng trái", (math.pi / 2, 0.0)), ("nghiêng phải", (-math.pi / 2, 0.0)), ("ngẫu nhiên", None)]
        kind = torch.arange(n, device=dev) % len(poses)
        quats = []
        g = torch.Generator().manual_seed(0)
        for i in range(n):
            rp = poses[int(kind[i])][1]
            if rp is None:
                rp = tuple(float(v) for v in (torch.rand(2, generator=g) * 2 - 1) * math.pi)
            yaw = float(torch.rand(1, generator=g) * 2 * math.pi)
            quats.append(quat_xyzw(rp[0], rp[1], yaw))
        pose = robot.data.default_root_pose.torch.clone()
        pose[:, :3] += u.scene.env_origins
        pose[:, 2] = u.scene.env_origins[:, 2] + 0.07
        pose[:, 3:7] = torch.tensor(quats, device=dev)
        ids = torch.arange(n, device=dev)
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
        # Để vịt rơi xuống sàn và nằm yên 1 s (không có lệnh khớp mới).
        hold = robot.data.default_joint_pos.torch.clone() * 0.0
        for _ in range(int(1.0 / dt)):
            obs, *_ = env.step(hold)
        start_tilt = tilt_deg(robot).clone()
        up_since = torch.full((n,), -1.0, device=dev)
        got_up = torch.full((n,), float("nan"), device=dev)
        steps = int(args.seconds / dt)
        for k in range(steps):
            o = obs["policy"].clone()
            o[:, CMD] = 0.0
            with torch.no_grad():
                obs, *_ = env.step(velstand(o))
            t = (k + 1) * dt
            z = robot.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]
            up = (tilt_deg(robot) < 20.0) & (z > 0.09)
            up_since = torch.where(up, torch.where(up_since < 0, torch.full_like(up_since, t), up_since), torch.full_like(up_since, -1.0))
            stable = up & (t - up_since >= 1.0) & torch.isnan(got_up)
            got_up = torch.where(stable, up_since, got_up)
        print(f"[POLLEN] đứng dậy: {n} vịt, {args.seconds:.0f} s, lệnh 0 (đứng yên đủ 1 s mới tính)", flush=True)
        for j, (name, _) in enumerate(poses):
            sel = kind == j
            ok = ~torch.isnan(got_up[sel])
            times = got_up[sel][ok]
            print(f"[POLLEN]   {name:13s}: nghiêng lúc đầu {float(start_tilt[sel].mean()):5.1f}°, đứng dậy "
                  f"{int(ok.sum())}/{int(sel.sum())}, thời gian TB {float(times.mean()) if len(times) else float('nan'):.2f} s, "
                  f"chậm nhất {float(times.max()) if len(times) else float('nan'):.2f} s", flush=True)
        ok = ~torch.isnan(got_up)
        print(f"[POLLEN] tổng: {int(ok.sum())}/{n} = {float(ok.float().mean()):.0%}", flush=True)

    else:
        ball = u.scene["ball"]
        kick = load(f"ball_kick_{args.foot}.pt")
        ids = torch.arange(n, device=dev)
        # Đứng yên 2 s cho ổn định.
        for _ in range(int(2.0 / dt)):
            o = obs["policy"].clone()
            o[:, CMD] = 0.0
            with torch.no_grad():
                obs, *_ = env.step(velstand(o))
        # Đặt bóng trước chân sút trong khung hướng của vịt (như lúc Pollen train: x 0,09 m, y ±0,042 m).
        yaw = robot.data.heading_w.torch
        side = -0.042 if args.foot == "right" else 0.042
        off = torch.tensor((0.09, side), device=dev)
        c, s = torch.cos(yaw), torch.sin(yaw)
        xy = robot.data.root_pos_w.torch[:, :2] + torch.stack((c * off[0] - s * off[1], s * off[0] + c * off[1]), -1)
        bpose = torch.zeros(n, 7, device=dev)
        bpose[:, :2] = xy
        bpose[:, 2] = u.scene.env_origins[:, 2] + 0.036
        bpose[:, 6] = 1.0
        ball.write_root_pose_to_sim_index(root_pose=bpose, env_ids=ids)
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
        start = xy.clone()
        heading = torch.stack((c, s), -1)
        peak = torch.zeros(n, device=dev)
        peak_dir = torch.zeros(n, device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        total = int((args.kick_s + 2.5) / dt)
        for k in range(total):
            o = obs["policy"].clone()
            o[:, CMD] = 0.0
            policy = kick if k * dt < args.kick_s else velstand
            with torch.no_grad():
                obs, *_ = env.step(policy(o))
            v = ball.data.root_lin_vel_w.torch[:, :2]
            sp = torch.linalg.norm(v, dim=-1)
            better = sp > peak
            ang = torch.rad2deg(torch.atan2(heading[:, 0] * v[:, 1] - heading[:, 1] * v[:, 0], (heading * v).sum(-1)))
            peak_dir = torch.where(better, ang, peak_dir)
            peak = torch.maximum(peak, sp)
            fell |= tilt_deg(robot) > 70.0
        disp = ball.data.root_pos_w.torch[:, :2] - start
        fwd = (disp * heading).sum(-1)
        kicked = peak > 0.2
        print(f"[POLLEN] sút chân {args.foot}, {args.kick_s:.1f} s sút rồi 2,5 s đứng: {n} vịt", flush=True)
        print(f"[POLLEN]   bóng bay ra (>0,2 m/s): {int(kicked.sum())}/{n}; tốc độ đỉnh TB {float(peak.mean()):.2f} m/s "
              f"(min {float(peak.min()):.2f}, max {float(peak.max()):.2f})", flush=True)
        print(f"[POLLEN]   hướng bóng so với hướng vịt: TB {float(peak_dir[kicked].mean()) if kicked.any() else float('nan'):+.1f}°, "
              f"lệch tuyệt đối TB {float(peak_dir[kicked].abs().mean()) if kicked.any() else float('nan'):.1f}°", flush=True)
        print(f"[POLLEN]   bóng đi về trước TB {float(fwd.mean()):.2f} m sau 3 s; vịt ngã {int(fell.sum())}/{n}", flush=True)
    env.close()
