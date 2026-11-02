"""Train hoặc xem trận MicroDuck N đấu N (soccer_team.py), mỗi vịt một não (mappo.py).

  python team.py train --team_size 2 --num_envs 512 --max_iterations 2000 [--resume <model.pt>]
  python team.py play  --team_size 2 --checkpoint <model.pt> --opponent scripted --num_envs 1 --viz viser
  python team.py eval  --team_size 2 --checkpoint <model.pt> --opponent scripted --num_envs 64 --seconds 300
num_envs là số trận. --opponent (đội B): scripted (chiến thuật viết tay), self (chính não đó), hoặc đường dẫn
checkpoint khác. eval in thắng / hoà / thua, số bàn, tỷ lệ thời gian bóng ở góc / đứng yên, số vịt quanh bóng,
số lần đá lại mỗi kiểu. Log ở logs/mappo/microduck_soccer_<N>v<N>/.
"""

import argparse
import os
import sys
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["train", "play", "eval"])
parser.add_argument("--team_size", type=int, default=2)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", default=None, help="checkpoint để train tiếp")
parser.add_argument("--checkpoint", default=None, help="checkpoint để xem / đo")
parser.add_argument("--opponent", default="scripted", help="play/eval: scripted | self | <checkpoint>")
parser.add_argument("--seconds", type=float, default=300.0, help="eval: thời gian mô phỏng")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
parser.set_defaults(device=None)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
os.environ["SOCCER_TEAM_SIZE"] = str(args.team_size)
if args.mode == "play":
    os.environ["SOCCER_FIELD_VISUALS"] = "1"
if args.mode != "play" and not args.visualizer:
    args.visualizer = ["none"]

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from isaaclab_tasks.contrib.microduck_course.mappo import (  # noqa: E402
    LATEST,
    SCRIPTED,
    MappoCfg,
    MappoTrainer,
    load_brains,
)
from isaaclab_tasks.contrib.microduck_course.soccer_team import TASK, TeamVecEnv  # noqa: E402

env_cfg, _ = resolve_task_config(TASK, "")
if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
args.device = env_cfg.sim.device
cfg = MappoCfg()
if args.max_iterations is not None:
    cfg.max_iterations = args.max_iterations
torch.manual_seed(args.seed)


def set_opponent(trainer: MappoTrainer, spec: str):
    """Đội B cho play / eval: mọi trận cùng một đối thủ."""
    if spec == "scripted":
        trainer.opp[:] = SCRIPTED
    elif spec == "self":
        trainer.opp[:] = LATEST
    else:
        trainer.pool = [load_brains(spec, trainer.n, trainer.obs_dim, trainer.act_dim, trainer.device)]
        trainer.opp[:] = 0


def team_actions(trainer: MappoTrainer, obs):
    obs_a, obs_b, _ = trainer._split(obs)
    act_a = trainer.brains.act(obs_a, deterministic=True)
    act_b = trainer.opponent_actions(obs_b, deterministic=True)
    return torch.cat((act_a, act_b), 1).reshape(-1, trainer.act_dim)


with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    vec = TeamVecEnv(env, args.team_size)
    name = f"microduck_soccer_{args.team_size}v{args.team_size}"
    print(f"[TEAM] {args.team_size} đấu {args.team_size}: {env.num_envs} trận, {vec.num_envs} vịt, mỗi vịt đội A một não", flush=True)
    if args.mode == "train":
        log_dir = os.path.abspath(os.path.join("logs", "mappo", name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        trainer = MappoTrainer(vec, cfg, log_dir=log_dir)
        if args.resume:
            trainer.load(args.resume)
        print(f"[TEAM] log: {log_dir}", flush=True)
        trainer.learn(cfg.max_iterations)
    else:
        trainer = MappoTrainer(vec, cfg)
        trainer.load(args.checkpoint, with_optimizer=False)
        set_opponent(trainer, args.opponent)
        obs = vec.get_observations()
        if args.mode == "eval":
            steps = int(args.seconds / env.step_dt)
            results = torch.zeros(3)
            for _ in range(steps):
                obs, _, dones, _ = vec.step(team_actions(trainer, obs))
                done = dones.view(env.num_envs, -1)[:, 0].bool()
                if bool(done.any()):
                    s = vec.final_score[done]
                    o = torch.sign(s[:, 0] - s[:, 1])
                    results += torch.stack(((o > 0).sum(), (o == 0).sum(), (o < 0).sum())).float().cpu()
            t = dict(zip(TeamVecEnv.STAT_NAMES, vec.totals.tolist()))
            live = max(t["steps"], 1.0)
            minutes = t["steps"] * env.step_dt / 60.0
            played = max(float(results.sum()), 1.0)
            print(f"[EVAL] {args.checkpoint} vs {args.opponent}", flush=True)
            print(
                f"[EVAL] {int(results.sum())} trận đủ 60 s: thắng {results[0] / played:.0%}, hoà {results[1] / played:.0%}, "
                f"thua {results[2] / played:.0%} | bàn A {int(vec.goals[0])} B {int(vec.goals[1])} "
                f"= {float(vec.goals.sum()) / max(minutes, 1e-6):.2f} bàn/phút | bóng ở góc {t['corner'] / live:.1%} | "
                f"bóng đứng yên {t['stalled'] / live:.1%} | vịt quanh bóng {t['crowd'] / live:.2f} | "
                f"≥3 vịt quanh bóng {t['crowd3'] / live:.1%}",
                flush=True,
            )
            print(
                "[EVAL] mỗi phút: " + ", ".join(f"{k} {t[k] / max(minutes, 1e-6):.2f}" for k in TeamVecEnv.EVENTS),
                flush=True,
            )
        else:
            score = [0, 0]
            while env.sim.is_running():
                obs, _, dones, _ = vec.step(team_actions(trainer, obs))
                goals = [int(g) for g in vec.goals]
                if goals != score:
                    print(f"[TEAM] bàn thắng! tỷ số trận {int(vec.match_score[0, 0])} - {int(vec.match_score[0, 1])}", flush=True)
                if bool(dones.any()):
                    s = vec.final_score[0]
                    print(f"[TEAM] hết trận: A {int(s[0])} - {int(s[1])} B ({args.opponent})", flush=True)
                score = goals
    env.close()
