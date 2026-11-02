"""MAPPO cho trận MicroDuck N đấu N (TeamVecEnv): mỗi vịt đội A một bộ não (actor) riêng, một critic chung nhìn cả
đội (chỉ dùng lúc train). Đội B do đối thủ điều khiển, chọn lại mỗi trận:
  - một bản não cũ của đội A trong kho (cứ snapshot_every vòng thêm một bản),
  - bản mới nhất (tự đá với mình),
  - đội chiến thuật viết tay (scripted_team): người gần bóng nhất vòng ra sau bóng đẩy về khung thành, người còn
    lại bọc lót giữa bóng và khung thành nhà.
Đội B nhìn sân lật ngược nên dùng thẳng được não của đội A: vịt thứ j đội B dùng não thứ j của bản được chọn.
Chỉ dữ liệu đội A được học.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

SCRIPTED, LATEST = -2, -1  # mã đối thủ; >= 0 là chỉ số trong kho


@dataclass
class MappoCfg:
    steps_per_iter: int = 32
    max_iterations: int = 2000
    hidden: tuple[int, ...] = (256, 128, 64)
    init_std: float = 0.8
    learning_rate: float = 1e-3
    desired_kl: float = 0.01  # lr thích nghi như rsl_rl
    epochs: int = 5
    minibatches: int = 4
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    snapshot_every: int = 50
    pool_size: int = 20  # đầy thì bỏ bản cũ nhất (giữ bản đầu tiên)
    p_latest: float = 0.2
    p_scripted: float = 0.1  # còn lại: bản cũ trong kho (kho rỗng thì chia đều cho hai loại trên)
    save_every: int = 100


# --- mạng ------------------------------------------------------------------------------------


class Normalizer(nn.Module):
    """Chuẩn hoá theo trung bình / phương sai chạy."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        delta = mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + var * n + delta.square() * self.count * n / total) / total
        self.count = total

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-5.0, 5.0)


def mlp(inp: int, out: int, hidden) -> nn.Sequential:
    layers, last = [], inp
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden, init_std: float):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))

    def dist(self, x) -> Normal:
        mean = self.net(x)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class Brains(nn.Module):
    """Não của một đội: N actor và bộ chuẩn hoá quan sát của chúng (đây là thứ được cất vào kho đối thủ)."""

    def __init__(self, n: int, obs_dim: int, act_dim: int, hidden, init_std: float):
        super().__init__()
        self.obs_norm = Normalizer(obs_dim)
        self.actors = nn.ModuleList(Actor(obs_dim, act_dim, hidden, init_std) for _ in range(n))

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """obs (B, N, D) chưa chuẩn hoá → hành động (B, N, A)."""
        x = self.obs_norm(obs)
        out = []
        for j, actor in enumerate(self.actors):
            d = actor.dist(x[:, j])
            out.append(d.mean if deterministic else d.sample())
        return torch.stack(out, dim=1)


# --- đối thủ viết tay -------------------------------------------------------------------------


def scripted_team(obs: torch.Tensor, n: int, act_dim: int = 4) -> torch.Tensor:
    """Đội chiến thuật viết tay trên quan sát của TeamVecEnv (B, N, D) → hành động (B, N, act_dim).

    Quan sát (khung thân): 6:8 bóng, 10:12 khung thành đối phương, 12:14 khung thành nhà, 14:14+2(N-1) đồng đội.
    Người lấy bóng vòng ra sau bóng, quay mặt về khung thành; bóng nằm trước chân (như lúc Pollen train cú sút:
    x ≈ 0,09 m, |y| ≈ 0,04 m) và đã nhìn về khung thành thì sút (hành động thứ 4 = 1).
    """
    o = obs.reshape(-1, obs.shape[-1])
    ball, goal, home = o[:, 6:8], o[:, 10:12], o[:, 12:14]
    striker = torch.ones(o.shape[0], dtype=torch.bool, device=o.device)
    if n > 1:
        mates = o[:, 14 : 14 + 2 * (n - 1)].reshape(o.shape[0], n - 1, 2)
        striker = (torch.linalg.norm(ball, dim=-1)[:, None] <= torch.linalg.norm(mates - ball[:, None], dim=-1)).all(-1)

    def unit(v):
        return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)

    to_goal = unit(goal - ball)
    left = torch.stack((-to_goal[:, 1], to_goal[:, 0]), dim=-1)
    # Chỗ đứng để bóng nằm trước chân phải khi nhìn về khung thành: bóng ở (0,09; -0,042) trong khung thân.
    behind = ball - 0.09 * to_goal + 0.042 * left
    cover = ball + 0.6 * unit(home - ball)
    target = torch.where(striker[:, None], behind, cover)
    near = torch.linalg.norm(ball, dim=-1) < 0.3
    # Xa bóng thì nhìn bóng; gần bóng thì người lấy bóng quay mặt theo hướng sút (bóng → khung thành).
    face = torch.where((striker & near)[:, None], to_goal, ball)
    yaw = torch.atan2(face[:, 1], face[:, 0])
    # Đi ngang lệnh mạnh hơn: velstand gần như không bước ngang với lệnh nhỏ.
    a = torch.stack((target[:, 0] / 0.2, target[:, 1] / 0.05, yaw / 0.5), dim=-1).clamp(-1.0, 1.0)
    if act_dim > 3:
        in_zone = (ball[:, 0] > 0.04) & (ball[:, 0] < 0.16) & (ball[:, 1].abs() < 0.1)
        aimed = torch.atan2(to_goal[:, 1], to_goal[:, 0]).abs() < 0.3
        kick = torch.where(striker & in_zone & aimed, 1.0, -1.0)
        a = torch.cat((a, kick[:, None]), dim=-1)
    return a.view(obs.shape[0], n, act_dim)


# --- train -----------------------------------------------------------------------------------


class MappoTrainer:
    def __init__(self, vec, cfg: MappoCfg, log_dir: str | None = None):
        self.vec, self.cfg = vec, cfg
        self.n, self.m, self.device = vec.n, vec.matches, vec.device
        obs = vec.get_observations()
        self.obs_dim = obs["policy"].shape[-1]
        self.act_dim = vec.num_actions
        self.extra_dim = obs["critic"].shape[-1] - self.obs_dim  # trạng thái bóng trong khung sân
        self.brains = Brains(self.n, self.obs_dim, self.act_dim, cfg.hidden, cfg.init_std).to(self.device)
        self.extra_norm = Normalizer(self.extra_dim).to(self.device)
        # Critic chung: quan sát của cả đội A + trạng thái bóng + mã vịt (one-hot) → giá trị của vịt đó.
        self.critic = mlp(self.n * self.obs_dim + self.extra_dim + self.n, 1, cfg.hidden).to(self.device)
        self.params = list(self.brains.actors.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.learning_rate)
        self.lr = cfg.learning_rate
        self.pool: list[Brains] = []
        self.opp = torch.full((self.m,), LATEST, dtype=torch.long, device=self.device)
        self.iteration = 0
        self.log_dir = log_dir
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

    # --- đối thủ ---

    def _sample_opponents(self, ids):
        c = self.cfg
        u = torch.rand(len(ids), device=self.device)
        if self.pool:
            opp = torch.randint(0, len(self.pool), (len(ids),), device=self.device)
            opp = torch.where(u < c.p_latest, LATEST, opp)
            opp = torch.where((u >= c.p_latest) & (u < c.p_latest + c.p_scripted), SCRIPTED, opp)
        else:
            opp = torch.where(u < c.p_latest / (c.p_latest + c.p_scripted), LATEST, SCRIPTED)
        self.opp[ids] = opp

    def _snapshot(self):
        snap = copy.deepcopy(self.brains).eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        if len(self.pool) >= self.cfg.pool_size:
            del self.pool[1]
            # Các trận đang đá với bản vừa bỏ chuyển sang bản mới nhất; chỉ số phía sau lùi một.
            self.opp = torch.where(self.opp == 1, LATEST, torch.where(self.opp > 1, self.opp - 1, self.opp))
        self.pool.append(snap)

    def opponent_actions(self, obs_b, deterministic=False):
        """obs_b (M, N, D) quan sát đội B → hành động theo đối thủ của từng trận."""
        act = torch.zeros(self.m, self.n, self.act_dim, device=self.device)
        for code in torch.unique(self.opp).tolist():
            sel = (self.opp == code).nonzero(as_tuple=False).squeeze(-1)
            if code == SCRIPTED:
                act[sel] = scripted_team(obs_b[sel], self.n, self.act_dim)
            else:
                brains = self.brains if code == LATEST else self.pool[code]
                act[sel] = brains.act(obs_b[sel], deterministic)
        return act

    # --- critic ---

    def _critic_input(self, x_a, extra):
        """x_a (B, N, D) quan sát đã chuẩn hoá của đội A, extra (B, E) → (B, N, N*D + E + N)."""
        b = x_a.shape[0]
        team = x_a.reshape(b, 1, -1).expand(b, self.n, -1)
        ex = self.extra_norm(extra)[:, None].expand(b, self.n, -1)
        ids = torch.eye(self.n, device=self.device)[None].expand(b, -1, -1)
        return torch.cat((team, ex, ids), dim=-1)

    def _split(self, obs):
        pol = obs["policy"].view(self.m, 2 * self.n, -1)
        extra = obs["critic"].view(self.m, 2 * self.n, -1)[:, 0, self.obs_dim :]  # khung đội A
        return pol[:, : self.n], pol[:, self.n :], extra

    # --- vòng train ---

    def learn(self, iterations: int):
        c, n, m, dev = self.cfg, self.n, self.m, self.device
        T = c.steps_per_iter
        self._sample_opponents(torch.arange(m, device=dev))
        env = self.vec.env
        env.episode_length_buf[:] = torch.randint_like(env.episode_length_buf, int(self.vec.max_episode_length))
        obs = self.vec.get_observations()
        D, E = self.obs_dim, self.extra_dim
        buf = {
            "x": torch.zeros(T, m, n, D, device=dev),
            "cin": torch.zeros(T, m, n, n * D + E + n, device=dev),
            "act": torch.zeros(T, m, n, self.act_dim, device=dev),
            "logp": torch.zeros(T, m, n, device=dev),
            "mu": torch.zeros(T, m, n, self.act_dim, device=dev),
            "sigma": torch.zeros(T, m, n, self.act_dim, device=dev),
            "value": torch.zeros(T, m, n, device=dev),
            "rew": torch.zeros(T, m, n, device=dev),
            "done": torch.zeros(T, m, device=dev),
        }
        start_it = self.iteration
        while self.iteration < start_it + iterations:
            t0 = time.time()
            if self.iteration % c.snapshot_every == 0:
                self._snapshot()
            logs = defaultdict(float)
            results = defaultdict(lambda: torch.zeros(3, device=dev))  # thắng / hoà / thua theo loại đối thủ
            with torch.no_grad():
                for t in range(T):
                    obs_a, obs_b, extra = self._split(obs)
                    self.brains.obs_norm.update(obs_a)
                    self.extra_norm.update(extra)
                    x = self.brains.obs_norm(obs_a)
                    acts, logps, mus, sigmas = [], [], [], []
                    for j, actor in enumerate(self.brains.actors):
                        d = actor.dist(x[:, j])
                        a = d.sample()
                        acts.append(a)
                        logps.append(d.log_prob(a).sum(-1))
                        mus.append(d.mean)
                        sigmas.append(d.stddev)
                    act_a = torch.stack(acts, 1)
                    cin = self._critic_input(x, extra)
                    value = self.critic(cin).squeeze(-1)
                    act_b = self.opponent_actions(obs_b)
                    obs, rew, dones, extras = self.vec.step(torch.cat((act_a, act_b), 1).reshape(-1, self.act_dim))
                    rew = rew.view(m, 2 * n)[:, :n]
                    done = dones.view(m, 2 * n)[:, 0].bool()
                    time_out = extras["time_outs"].view(m, 2 * n)[:, 0].float()
                    # Trận chỉ kết thúc khi hết giờ: cộng giá trị ước lượng của trạng thái cuối (như rsl_rl).
                    rew = rew + c.gamma * value * time_out[:, None]
                    for key, val in (("x", x), ("cin", cin), ("act", act_a), ("logp", torch.stack(logps, 1)),
                                     ("mu", torch.stack(mus, 1)), ("sigma", torch.stack(sigmas, 1)),
                                     ("value", value), ("rew", rew), ("done", done.float())):
                        buf[key][t] = val
                    for key, val in extras["log"].items():
                        logs[key] += float(val) / T
                    logs["reward_per_step"] += float(rew.mean()) / T
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if len(ids):
                        score = self.vec.final_score[ids]
                        outcome = torch.sign(score[:, 0] - score[:, 1])  # +1 A thắng
                        for code in torch.unique(self.opp[ids]).tolist():
                            sel = self.opp[ids] == code
                            name = "scripted" if code == SCRIPTED else "latest" if code == LATEST else "pool"
                            o = outcome[sel]
                            results[name] += torch.stack(((o > 0).sum(), (o == 0).sum(), (o < 0).sum())).float()
                        self._sample_opponents(ids)
                obs_a, _, extra = self._split(obs)
                last_value = self.critic(self._critic_input(self.brains.obs_norm(obs_a), extra)).squeeze(-1)

            adv, ret = self._gae(buf, last_value)
            stats = self._update(buf, adv, ret)
            self.iteration += 1
            stats.update(logs)
            for name, r in results.items():
                total = float(r.sum())
                if total > 0:
                    stats[f"Match/{name}_win"] = float(r[0]) / total
                    stats[f"Match/{name}_draw"] = float(r[1]) / total
                    stats[f"Match/{name}_matches"] = total
            stats["Train/lr"] = self.lr
            stats["Train/pool_size"] = len(self.pool)
            stats["Train/iteration_time"] = time.time() - t0
            self._log(stats)
            if self.log_dir and (self.iteration % c.save_every == 0 or self.iteration == start_it + iterations):
                self.save(os.path.join(self.log_dir, f"model_{self.iteration}.pt"))

    def _gae(self, buf, last_value):
        c = self.cfg
        T = buf["rew"].shape[0]
        adv = torch.zeros_like(buf["rew"])
        running = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else buf["value"][t + 1]
            alive = 1.0 - buf["done"][t][:, None]
            delta = buf["rew"][t] + c.gamma * next_value * alive - buf["value"][t]
            running = delta + c.gamma * c.lam * alive * running
            adv[t] = running
        return adv, adv + buf["value"]

    def _update(self, buf, adv, ret):
        c, n = self.cfg, self.n
        flat = {k: v.flatten(0, 1) for k, v in buf.items() if k != "done"}
        adv = adv.flatten(0, 1)
        adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)  # chuẩn hoá theo từng vịt
        ret = ret.flatten(0, 1)
        size = adv.shape[0]
        mb = size // c.minibatches
        sums = defaultdict(float)
        count = 0
        for _ in range(c.epochs):
            perm = torch.randperm(size, device=self.device)
            for k in range(c.minibatches):
                idx = perm[k * mb : (k + 1) * mb]
                x, a = flat["x"][idx], flat["act"][idx]
                policy_loss, entropy, kl = 0.0, 0.0, 0.0
                for j, actor in enumerate(self.brains.actors):
                    d = actor.dist(x[:, j])
                    logp = d.log_prob(a[:, j]).sum(-1)
                    ratio = torch.exp(logp - flat["logp"][idx, j])
                    aj = adv[idx, j]
                    surr = torch.max(-aj * ratio, -aj * ratio.clamp(1 - c.clip, 1 + c.clip))
                    policy_loss = policy_loss + surr.mean()
                    entropy = entropy + d.entropy().sum(-1).mean()
                    with torch.no_grad():
                        mu0, s0 = flat["mu"][idx, j], flat["sigma"][idx, j]
                        mu1, s1 = d.mean, d.stddev
                        kl = kl + (torch.log(s1 / s0) + (s0.square() + (mu0 - mu1).square()) / (2 * s1.square()) - 0.5).sum(-1).mean()
                kl = kl / n
                if c.desired_kl > 0:
                    if kl > 2 * c.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0 < kl < c.desired_kl / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                value = self.critic(flat["cin"][idx]).squeeze(-1)
                v_old = flat["value"][idx]
                v_clip = v_old + (value - v_old).clamp(-c.clip, c.clip)
                value_loss = torch.max((value - ret[idx]).square(), (v_clip - ret[idx]).square()).mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, c.max_grad_norm)
                self.opt.step()
                sums["Loss/policy"] += policy_loss.item() / n
                sums["Loss/value"] += value_loss.item()
                sums["Loss/entropy"] += entropy.item() / n
                sums["Loss/kl"] += float(kl)
                count += 1
        out = {k: v / count for k, v in sums.items()}
        for j, actor in enumerate(self.brains.actors):
            out[f"Policy/std_duck{j}"] = float(actor.log_std.exp().mean())
        return out

    def _log(self, stats):
        if self.writer:
            for k, v in stats.items():
                self.writer.add_scalar(k, v, self.iteration)
        g = stats.get
        line = (
            f"[MAPPO] vòng {self.iteration} | {g('Train/iteration_time', 0):.1f} s | thưởng/bước {g('reward_per_step', 0):+.3f}"
            f" | bàn/phút {g('Soccer/goals_per_min', 0):.2f} | bóng ở góc {g('Soccer/ball_corner_frac', 0):.1%}"
            f" | đứng yên {g('Soccer/ball_stalled_frac', 0):.1%} | vịt quanh bóng {g('Soccer/ducks_near_ball', 0):.2f}"
            f" | sút/phút {g('Soccer/kick_per_min', 0):.2f} trúng khung {g('Soccer/shot_per_min', 0):.2f}"
            f" | chuyền/phút {g('Soccer/pass_per_min', 0):.2f} | ngã/phút {g('Soccer/fall_per_min', 0):.2f}"
            f" | ném vào/phút {g('Soccer/throw_in_per_min', 0):.2f}"
        )
        for name in ("scripted", "latest", "pool"):
            if f"Match/{name}_win" in stats:
                line += f" | thắng {name} {stats[f'Match/{name}_win']:.0%} ({int(stats[f'Match/{name}_matches'])} trận)"
        print(line + f" | lr {self.lr:.1e} | kho {len(self.pool)}", flush=True)

    # --- lưu / nạp ---

    def save(self, path):
        torch.save(
            {
                "iteration": self.iteration,
                "brains": self.brains.state_dict(),
                "critic": self.critic.state_dict(),
                "extra_norm": self.extra_norm.state_dict(),
                "opt": self.opt.state_dict(),
                "lr": self.lr,
                "pool": [p.state_dict() for p in self.pool],
            },
            path,
        )
        print(f"[MAPPO] lưu {path}", flush=True)

    def load(self, path, with_optimizer: bool = True):
        data = torch.load(path, map_location=self.device)
        self.brains.load_state_dict(data["brains"])
        self.critic.load_state_dict(data["critic"])
        self.extra_norm.load_state_dict(data["extra_norm"])
        self.iteration = data["iteration"]
        if with_optimizer:
            self.opt.load_state_dict(data["opt"])
            self.lr = data["lr"]
        self.pool = []
        for state in data["pool"]:
            snap = copy.deepcopy(self.brains)
            snap.load_state_dict(state)
            self.pool.append(snap.eval())
        print(f"[MAPPO] nạp {path} (vòng {self.iteration}, kho {len(self.pool)})", flush=True)


def load_brains(path, n, obs_dim, act_dim, device) -> Brains:
    """Não đội A trong một checkpoint (làm đối thủ hoặc để xem)."""
    data = torch.load(path, map_location=device)
    hidden = MappoCfg.hidden
    brains = Brains(n, obs_dim, act_dim, hidden, 1.0).to(device)
    brains.load_state_dict(data["brains"])
    return brains.eval()
