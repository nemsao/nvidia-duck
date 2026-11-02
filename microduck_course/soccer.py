"""Bộ chiến thuật đá bóng cho MicroDuck: thay lệnh vận tốc ngẫu nhiên bằng lệnh tính từ vị trí bóng và khung thành.

Bộ não đi bộ (RL) giữ nguyên; tầng này chỉ quyết định đi đâu, giống cách các đội RoboCup tách "dáng đi" và "chiến thuật":
  1. Chưa ở sau bóng: đi vòng tới điểm sau bóng trên đường bóng → khung thành (vòng qua bên nếu đang đứng trước bóng).
  2. Đã ở sau bóng, thẳng hàng: đi thẳng qua bóng để dẫn và đẩy bóng vào khung thành.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from ..microduck.mdp.commands import MicroDuckVelocityCommand, MicroDuckVelocityCommandCfg

GOALS_SCORED = [0]


class SoccerVelocityCommand(MicroDuckVelocityCommand):
    cfg: SoccerVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int] | slice):
        self.is_standing_env[env_ids] = False  # không bao giờ đứng yên

    def reset(self, env_ids: Sequence[int] | None = None):
        if os.environ.get("SOCCER_TRACE") and getattr(self._env, "termination_manager", None) is not None:
            tm = self._env.termination_manager
            why = [name for name in tm.active_terms if bool(tm.get_term(name)[0])]
            steps = int(self._env.episode_length_buf[0])
            print(f"[TRACE] hết lượt sau {steps * self._env.step_dt:.1f} s, lý do: {why}", flush=True)
        return super().reset(env_ids)

    def _update_command(self):
        origins = self._env.scene.env_origins[:, :2]
        robot = self.robot.data.root_pos_w.torch[:, :2] - origins
        yaw = self.robot.data.heading_w.torch
        ball = self._env.scene["ball"].data.root_pos_w.torch[:, :2] - origins
        goal = torch.tensor(self.cfg.goal_xy, device=self.device).expand_as(ball)

        to_goal = goal - ball
        d = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
        n = torch.stack((-d[:, 1], d[:, 0]), dim=-1)  # pháp tuyến trái của đường bóng → khung thành
        rel = robot - ball
        along = (rel * d).sum(-1)  # âm = đứng sau bóng
        side = (rel * n).sum(-1)

        behind = ball - d * self.cfg.behind_dist
        lined_up = (along < -0.6 * self.cfg.behind_dist) & (side.abs() < self.cfg.line_tolerance)
        in_front = along > -0.05
        # Đang đứng trước bóng: đi vòng sang bên mình đang đứng, tránh đá phản lưới.
        side_sign = torch.where(side >= 0, 1.0, -1.0).unsqueeze(-1)
        detour = ball - d * 0.22 + n * side_sign * 0.22
        target = torch.where(in_front.unsqueeze(-1), detour, behind)
        target = torch.where(lined_up.unsqueeze(-1), ball + d * 0.3, target)

        to_target = target - robot
        dist = torch.linalg.norm(to_target, dim=-1)
        face = torch.where(
            lined_up | (dist < 0.2), torch.atan2(d[:, 1], d[:, 0]), torch.atan2(to_target[:, 1], to_target[:, 0])
        )
        yaw_err = math_utils.wrap_to_pi(face - yaw)

        speed = torch.clamp(dist * 1.5, max=self.cfg.max_speed)
        speed = speed * torch.clamp(1.0 - yaw_err.abs() / 1.2, min=0.15)  # lệch hướng nhiều thì xoay trước
        world_v = to_target / dist.clamp_min(1e-6).unsqueeze(-1) * speed.unsqueeze(-1)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        vx = cos * world_v[:, 0] + sin * world_v[:, 1]
        vy = -sin * world_v[:, 0] + cos * world_v[:, 1]
        r = self.cfg.ranges
        self.vel_command_b[:, 0] = vx.clamp(*r.lin_vel_x)
        self.vel_command_b[:, 1] = vy.clamp(*r.lin_vel_y)
        self.vel_command_b[:, 2] = (2.0 * yaw_err).clamp(*r.ang_vel_z)

        self._trace_count = getattr(self, "_trace_count", 0) + 1
        if os.environ.get("SOCCER_TRACE") and self._trace_count % 100 == 0:
            mode = "đẩy" if bool(lined_up[0]) else ("vòng" if bool(in_front[0]) else "ra sau bóng")
            f = lambda v: f"({float(v[0]):+.2f}, {float(v[1]):+.2f})"  # noqa: E731
            vel = self.robot.data.root_lin_vel_b.torch[0]
            print(
                f"[TRACE] vịt {f(robot[0])} yaw {float(yaw[0]):+.2f} bóng {f(ball[0])} đích {f(target[0])} {mode} "
                f"lệnh {[round(float(x), 2) for x in self.vel_command_b[0]]} vận tốc thật {f(vel)}",
                flush=True,
            )


@configclass
class SoccerVelocityCommandCfg(MicroDuckVelocityCommandCfg):
    class_type: type = SoccerVelocityCommand
    goal_xy: tuple[float, float] = (1.6, 0.0)  # tâm khung thành tấn công (sau vạch vôi một chút)
    behind_dist: float = 0.2  # điểm chờ sau bóng (tâm bóng → tâm vịt)
    line_tolerance: float = 0.05  # lệch ngang tối đa so với đường bóng → khung thành để bắt đầu đẩy
    max_speed: float = 0.35


def ball_in_goal(env, goal_line_x: float, half_width: float):
    """Kết thúc lượt khi bóng qua vạch vôi giữa hai cột dọc (ghi bàn)."""
    ball = env.scene["ball"].data.root_pos_w.torch - env.scene.env_origins
    scored = (ball[:, 0] > goal_line_x) & (ball[:, 1].abs() < half_width)
    if bool(scored.any()):
        GOALS_SCORED[0] += int(scored.sum())
        print(f"[SOCCER] VÀO! tổng {GOALS_SCORED[0]} bàn", flush=True)
    return scored
