"""Hàm MDP cho bài MicroDuck học đá bóng (tầng chiến thuật học bằng RL, tầng đi bộ là chính sách đã train).

Mọi toạ độ tính trong khung sân của từng môi trường (gốc ở giữa sân, khung thành đối phương ở +x).
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager, SceneEntityCfg
from isaaclab.utils import configclass, instantiate

from isaaclab_tasks.contrib.microduck.mdp.observations import misaligned_imu


# --- môi trường ------------------------------------------------------------------------------


class HierarchicalRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv gọi apply_action ở mỗi bước vật lý.

    Khi mọi động cơ Newton chạy được bằng CUDA graph, Isaac Lab gộp cả vòng decimation vào một lần sim.step() và chỉ
    gọi apply_action một lần mỗi bước môi trường. Tầng đi bộ khi đó chỉ chạy 10 Hz thay vì 50 Hz và vịt ngã. Ở đây
    Newton chạy từng bước (vẫn bằng CUDA graph một bước), env tự lặp và gọi apply_action mỗi bước.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.sim.backend_manager.set_decimation(1)
        self._physics_handles_decimation = False


# --- tầng hành động ---------------------------------------------------------------------------


class imu_every_call(misaligned_imu):
    """IMU của MicroDuck, đọc mới ở mỗi lần gọi.

    Bản gốc trả lại số cũ khi bị gọi lại trong cùng env.common_step_counter; ở đây tầng đi bộ chạy 5 lần mỗi bước
    tầng trên nên 4/5 lần sẽ nhận góc nghiêng cũ và vịt ngã sau ~0,5 s.
    """

    def __call__(
        self,
        env,
        max_angle_deg: float,
        min_lag: int,
        max_lag: int,
        update_period: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        self._last_step = None
        return super().__call__(env, max_angle_deg, min_lag, max_lag, update_period, asset_cfg)


def _retarget_low_level(cfg: HighLevelVelocityActionCfg):
    """Cho tầng đi bộ điều khiển con vịt cfg.asset_name thay vì "robot" (trận nhiều vịt: robot_0, robot_1, ...).

    Quan sát của MicroDuck mặc định đọc "robot" và lấy encoder_bias từ khối hành động "joint_pos".
    """
    obs = cfg.low_level_observations

    def rename(value):
        return value.replace(name=cfg.asset_name) if isinstance(value, SceneEntityCfg) and value.name == "robot" else value

    for name in ("imu", "joint_pos", "joint_vel"):
        term = getattr(obs, name)
        term.params = {key: rename(value) for key, value in term.params.items()}
    obs.imu.params["asset_cfg"] = SceneEntityCfg(cfg.asset_name)
    obs.joint_pos.params["action_name"] = cfg.term_name
    cfg.low_level_actions = cfg.low_level_actions.replace(asset_name=cfg.asset_name)


class HighLevelVelocityAction(ActionTerm):
    """Hành động [-1, 1]^3 của chính sách đá bóng → lệnh (vx, vy, wz) cho chính sách đi bộ đóng băng.

    Giống PreTrainedPolicyAction của bài điều hướng, khác hai chỗ để hợp với quan sát của MicroDuck:
      - Khối này phải tên "joint_pos" và cho mượn encoder_bias của khối khớp bên trong: quan sát góc khớp và sự kiện
        ngẫu nhiên hoá encoder của MicroDuck đọc nó qua env.action_manager.get_term("joint_pos").
      - Bộ quan sát của tầng đi bộ dựng ở lần chạy đầu: lúc khởi tạo khối, env.action_manager chưa có.

    Bộ chọn kỹ năng (khi cfg.getup = True và / hoặc có cfg.kick_policy_paths), chạy ở nhịp tầng dưới:
      - ĐI: chính sách đi bộ với lệnh của tầng trên, hoặc lệnh ép (override_cmd) khi trọng tài cho chạy về vị trí;
      - NGÃ: nghiêng quá fall_tilt_deg → cùng chính sách đi bộ với mọi lệnh = 0 (velstand của Pollen tự đứng dậy),
        thẳng lại (< up_tilt_deg) đủ up_hold_s thì về ĐI;
      - SÚT: hành động thứ 4 > 0,5 khi đang ĐI → kick_left / kick_right (chân phía có bóng) trong kick_s, lệnh = 0,
        rồi về ĐI; phải chờ kick_cooldown_s mới sút tiếp.
    Các chính sách đều theo giao kèo 61 chiều của MicroDuck nên dùng chung một bộ quan sát tầng dưới.
    """

    WALK, DOWN, KICK = 0, 1, 2

    cfg: HighLevelVelocityActionCfg

    def __init__(self, cfg: HighLevelVelocityActionCfg, env):
        if cfg.asset_name != "robot":
            _retarget_low_level(cfg)
        super().__init__(cfg, env)
        self.policy = torch.jit.load(cfg.policy_path).to(env.device).eval()
        self.kicks = [torch.jit.load(p).to(env.device).eval() for p in (cfg.kick_policy_paths or ())]
        self._scale = torch.tensor(cfg.command_scale, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self._hl_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_term: ActionTerm = instantiate(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._joint_term.action_dim, device=self.device)
        # Bộ chọn kỹ năng.
        n = self.num_envs
        self.mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.mode_time = torch.zeros(n, device=self.device)  # s trong kỹ năng hiện tại
        self.up_time = torch.zeros(n, device=self.device)  # s đã thẳng lại liên tục khi đang NGÃ
        self.kick_foot = torch.zeros(n, dtype=torch.long, device=self.device)  # 0 trái, 1 phải
        self.kick_ready_at = torch.zeros(n, device=self.device)  # thời điểm (s) được sút tiếp
        self._kick_request = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.clock = torch.zeros(n, device=self.device)
        # Sự kiện cộng dồn tới khi TeamVecEnv đọc và xoá: bắt đầu sút, vừa ngã.
        self.kick_started = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fell = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.override_cmd = torch.zeros(n, 3, device=self.device)  # lệnh vật lý (m/s, rad/s)
        self.override_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._robot = env.scene[cfg.asset_name]
        self._ball = env.scene[cfg.kick_ball_name] if self.kicks and cfg.kick_ball_name else None

        def last_low_level_action():
            self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        cfg.low_level_observations.actions.func = lambda dummy_env: last_low_level_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._command
        cfg.low_level_observations.velocity_commands.params = dict()
        cfg.low_level_observations.imu.func = imu_every_call
        self._low_level_obs: ObservationManager | None = None
        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 4 if self.cfg.kick_policy_paths else 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._command

    @property
    def encoder_bias(self) -> torch.Tensor:
        return self._joint_term.encoder_bias

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._hl_command[:] = torch.clamp(actions[:, :3], -1.0, 1.0) * self._scale
        self._command[:] = self._hl_command
        if self.kicks:
            self._kick_request[:] = actions[:, 3] > 0.5

    def _update_skills(self, dt: float):
        """Chuyển kỹ năng ở nhịp tầng dưới; đặt lệnh cho chính sách đi bộ."""
        c = self.cfg
        self.clock += dt
        self.mode_time += dt
        tilt = torch.rad2deg(torch.acos((-self._robot.data.projected_gravity_b.torch[:, 2]).clamp(-1.0, 1.0)))
        if c.getup:
            fell = (self.mode != self.DOWN) & (tilt > c.fall_tilt_deg)
            self.fell |= fell
            self.mode = torch.where(fell, self.DOWN, self.mode)
            self.mode_time = torch.where(fell, 0.0, self.mode_time)
            self.up_time = torch.where(fell, 0.0, self.up_time)
            down = self.mode == self.DOWN
            self.up_time = torch.where(down & (tilt < c.up_tilt_deg), self.up_time + dt, 0.0)
            stood = down & (self.up_time >= c.up_hold_s)
            self.mode = torch.where(stood, self.WALK, self.mode)
            self.mode_time = torch.where(stood, 0.0, self.mode_time)
        if self.kicks:
            done = (self.mode == self.KICK) & (self.mode_time >= c.kick_s)
            self.mode = torch.where(done, self.WALK, self.mode)
            self.mode_time = torch.where(done, 0.0, self.mode_time)
            start = (
                self._kick_request & (self.mode == self.WALK) & ~self.override_mask & (self.clock >= self.kick_ready_at)
            )
            if bool(start.any()):
                # Chân phía có bóng (bóng bên trái thân → chân trái).
                rel = self._ball.data.root_pos_w.torch[:, :2] - self._robot.data.root_pos_w.torch[:, :2]
                yaw = self._robot.data.heading_w.torch
                side = -torch.sin(yaw) * rel[:, 0] + torch.cos(yaw) * rel[:, 1]
                self.kick_foot = torch.where(start, (side < 0).long(), self.kick_foot)
                self.mode = torch.where(start, self.KICK, self.mode)
                self.mode_time = torch.where(start, 0.0, self.mode_time)
                self.kick_ready_at = torch.where(start, self.clock + c.kick_s + c.kick_cooldown_s, self.kick_ready_at)
                self.kick_started |= start
            self._kick_request[:] = False  # mỗi lệnh sút chỉ dùng một lần
        walking = self.mode == self.WALK
        cmd = torch.where(self.override_mask[:, None], self.override_cmd, self._hl_command)
        self._command[:] = torch.where(walking[:, None], cmd, torch.zeros_like(cmd))

    def apply_actions(self):
        if self._low_level_obs is None:
            self._low_level_obs = ObservationManager({"ll_policy": self.cfg.low_level_observations}, self._env)
        if self._counter % self.cfg.low_level_decimation == 0:
            if self.cfg.getup or self.kicks:
                self._update_skills(self._env.physics_dt * self.cfg.low_level_decimation)
            # update_history=True: bộ đệm trễ (vận tốc khớp trễ 1 mẫu) phải tiến theo nhịp tầng dưới như lúc train.
            obs = self._low_level_obs.compute_group("ll_policy", update_history=True)
            with torch.inference_mode():
                if self.cfg.getup or self.kicks:
                    busy = (self.mode != self.WALK)[:, None]
                    obs = torch.where(busy & (torch.arange(obs.shape[-1], device=self.device) >= 48), 0.0, obs)
                actions = self.policy(obs)
                if self.kicks:
                    kick = torch.where((self.kick_foot == 0)[:, None], self.kicks[0](obs), self.kicks[1](obs))
                    actions = torch.where((self.mode == self.KICK)[:, None], kick, actions)
                self.low_level_actions[:] = actions
            self._joint_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._joint_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids=None):
        self._joint_term.reset(env_ids)
        if self._low_level_obs is not None:
            self._low_level_obs.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.mode[ids] = self.WALK
        self.mode_time[ids] = 0.0
        self.up_time[ids] = 0.0
        self.kick_ready_at[ids] = 0.0
        self.clock[ids] = 0.0
        self._kick_request[ids] = False
        self.override_mask[ids] = False


@configclass
class HighLevelVelocityActionCfg(ActionTermCfg):
    class_type: type = HighLevelVelocityAction
    asset_name: str = "robot"
    term_name: str = "joint_pos"  # tên khối này trong ActionsCfg (quan sát góc khớp tìm encoder_bias theo tên)
    policy_path: str = MISSING  # chính sách đi bộ TorchScript
    low_level_decimation: int = 4
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
    command_scale: tuple[float, float, float] = (0.4, 0.3, 1.0)  # giới hạn lệnh lúc train bộ não đi bộ
    # Bộ chọn kỹ năng (mặc định tắt: chỉ đi bộ như bài ghi bàn một vịt).
    getup: bool = False  # chính sách đi bộ tự đứng dậy khi lệnh = 0 (velstand của Pollen)
    fall_tilt_deg: float = 70.0
    up_tilt_deg: float = 20.0
    up_hold_s: float = 0.3
    kick_policy_paths: tuple[str, str] | None = None  # (chân trái, chân phải); có thì thêm hành động thứ 4 "sút"
    kick_ball_name: str | None = "ball"  # để chọn chân phía có bóng
    kick_s: float = 0.5  # thời lượng cú sút (manifest của Pollen)
    kick_cooldown_s: float = 0.5


# --- trạng thái chung ------------------------------------------------------------------------


def _robot(env):
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return pos, robot.data.heading_w.torch


def _ball(env):
    ball = env.scene["ball"]
    return ball.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2], ball.data.root_lin_vel_w.torch[:, :2]


def _to_body(vec, yaw):
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((cos * vec[:, 0] + sin * vec[:, 1], -sin * vec[:, 0] + cos * vec[:, 1]), dim=-1)


def _goal(env, goal_x):
    return torch.tensor((goal_x, 0.0), device=env.device).expand(env.num_envs, 2)


# --- quan sát --------------------------------------------------------------------------------


def robot_on_field(env):
    """Vị trí vịt trên sân và hướng (sin, cos)."""
    pos, yaw = _robot(env)
    return torch.cat((pos, torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1)), dim=-1)


def robot_lin_vel_b(env):
    return env.scene["robot"].data.root_lin_vel_b.torch[:, :2]


def ball_in_robot_frame(env):
    """Vị trí và vận tốc bóng nhìn từ vịt."""
    pos, yaw = _robot(env)
    ball, ball_vel = _ball(env)
    return torch.cat((_to_body(ball - pos, yaw), _to_body(ball_vel, yaw)), dim=-1)


def ball_on_field(env):
    ball, ball_vel = _ball(env)
    return torch.cat((ball, ball_vel), dim=-1)


def goal_in_robot_frame(env, goal_x: float):
    pos, yaw = _robot(env)
    return _to_body(_goal(env, goal_x) - pos, yaw)


# --- phần thưởng -----------------------------------------------------------------------------


def ball_velocity_to_goal(env, goal_x: float):
    """Tốc độ bóng hướng về khung thành đối phương (m/s, chặn trên 1)."""
    ball, ball_vel = _ball(env)
    to_goal = _goal(env, goal_x) - ball
    direction = to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)
    return (ball_vel * direction).sum(-1).clamp(-1.0, 1.0)


def near_ball(env, std: float):
    """Gần bóng thì được thưởng (định hướng lúc đầu, khi chưa biết dẫn bóng)."""
    pos, _ = _robot(env)
    ball, _ = _ball(env)
    return torch.exp(-torch.linalg.norm(ball - pos, dim=-1) / std)


def ball_goal_progress(env, goal_x: float, field_half_x: float):
    """Bóng càng gần khung thành đối phương càng tốt (0 ở vạch nhà, 1 ở vạch đối phương)."""
    ball, _ = _ball(env)
    return ((ball[:, 0] + field_half_x) / (2 * field_half_x)).clamp(0.0, 1.0) * (1.0 - ball[:, 1].abs().clamp(max=1.0))


# --- kết thúc lượt ---------------------------------------------------------------------------


def ball_crossed(env, goal_line_x: float, half_width: float):
    """Bóng qua vạch vôi giữa hai cột dọc. goal_line_x > 0: khung thành đối phương; < 0: khung thành nhà."""
    ball, _ = _ball(env)
    inside = ball[:, 1].abs() < half_width
    return inside & (ball[:, 0] > goal_line_x if goal_line_x > 0 else ball[:, 0] < goal_line_x)
