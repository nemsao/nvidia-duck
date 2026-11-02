"""Bài RL 2: hai đội MicroDuck N đấu N; mỗi vịt đội A một bộ não chiến thuật (mappo.py).

Môi trường Isaac Lab (MicroDuckSoccerTeamEnvCfg) lo cảnh: 2N vịt thân va chạm toàn phần, một bóng Ø7 cm / 15 g, sân
theo cỡ đội. Mỗi vịt có bộ chọn kỹ năng riêng (khối joint_pos_i, soccer_rl.HighLevelVelocityAction) chạy các não
của Pollen Robotics: velstand (đi bộ, tự đứng dậy khi ngã) và kick_left / kick_right (sút thật bằng chân).
Luật (TeamRulesCfg, TeamVecEnv làm trọng tài):
  - một trận 60 s; bóng qua vạch vôi giữa hai cột là bàn thắng; bóng về giữa sân, các vịt TỰ CHẠY về vị trí đội hình
    (tối đa return_s, vịt chưa tới mới bị dịch chuyển), rồi đội thủng lưới giao bóng;
  - bóng ra khỏi sân (biên dọc hoặc cuối sân, ngoài khung thành): "người vô hình" nhặt bóng ném bay vào một điểm ngẫu
    nhiên trong sân (bóng không bị dịch chuyển, chỉ được gán vận tốc bay);
  - vịt ngã tự đứng dậy; nằm quá getup_timeout_s mới bị phạt và dựng lại ở vị trí đội hình;
  - đầu trận vịt đứng theo đội hình (env reset).
TeamVecEnv biến mỗi trận thành 2N "người chơi": quan sát, phần thưởng tính riêng từng vịt; đội B nhìn sân lật ngược
nên cả hai đội đều tấn công về phía +x và dùng chung được não.
Số vịt mỗi đội lấy từ biến môi trường SOCCER_TEAM_SIZE (mặc định 2).
"""

from __future__ import annotations

import copy
import math
import os

import torch
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from rsl_rl.env import VecEnv

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_CFG

from . import soccer_rl
from .soccer_env_cfg import ball_cfg, field_layout
from .soccer_score_env_cfg import LOW_LEVEL

TASK = "IsaacContrib-Soccer-Team-MicroDuck"
MAX_TEAM = 5
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")
VELSTAND = os.path.join(POLICY_DIR, "velstand.pt")
KICKS = (os.path.join(POLICY_DIR, "ball_kick_left.pt"), os.path.join(POLICY_DIR, "ball_kick_right.pt"))
BALL_R, BALL_MASS = 0.035, 0.015  # bóng Pollen dùng để train kỹ năng sút

# Đội hình (tỷ lệ theo nửa sân) của đội tấn công về +x; đội kia đối xứng qua tâm sân.
FORMATIONS = {
    1: [(-0.4, 0.0)],
    2: [(-0.3, 0.3), (-0.6, -0.3)],
    3: [(-0.3, 0.3), (-0.3, -0.3), (-0.65, 0.0)],
    4: [(-0.3, 0.3), (-0.3, -0.3), (-0.65, 0.5), (-0.65, -0.5)],
    5: [(-0.25, 0.3), (-0.25, -0.3), (-0.5, 0.6), (-0.5, -0.6), (-0.78, 0.0)],
}


@configclass
class TeamRewardCfg:
    """Trọng số phần thưởng của từng vịt (TeamVecEnv)."""

    goal: float = 10.0  # đội mình ghi bàn
    conceded: float = -3.0  # đội mình thủng lưới (nhỏ: trừ nặng thì vịt thà giữ bóng kẹt ở góc còn hơn)
    pass_completed: float = 1.0  # cho vịt sút khi đồng đội chạm bóng tiếp theo trong pass_window_s
    shot_on_target: float = 0.5  # cho vịt sút khi đường bóng hướng vào khung thành đối phương
    kick_attempt: float = -0.05  # mỗi lần sút (để không sút liên tục khi không có bóng)
    unrecovered_fall: float = -2.0  # nằm quá getup_timeout_s mà chưa tự đứng dậy
    ball_to_goal: float = 0.1  # vận tốc bóng về khung thành đối phương (m/s, chặn ±1), cả đội
    ball_to_goal_toucher: float = 0.3  # như trên, thêm cho riêng vịt chạm bóng cuối nếu nó thuộc đội mình
    # Vai trò theo tình huống: mỗi đội vịt gần bóng nhất là người lấy bóng, các vịt còn lại hỗ trợ.
    near_ball: float = 0.05  # người lấy bóng áp sát bóng
    near_ball_std: float = 0.3  # m
    support_pos: float = 0.05  # vịt hỗ trợ đứng ở chỗ bọc lót: giữa bóng và khung thành nhà, cách bóng support_dist
    support_dist: float = 0.6  # m
    support_std: float = 0.3  # m
    support_near_ball: float = -0.05  # vịt hỗ trợ lao vào sát bóng (trong crowd_radius)
    mate_close: float = -0.1  # hai đồng đội đứng gần nhau hơn mate_dist (mỗi con)
    mate_dist: float = 0.35  # m
    stalled: float = -0.02  # bóng đứng yên khi đang đá tự do (cả hai đội, để không bên nào lợi khi giữ bóng kẹt)


@configclass
class TeamRulesCfg:
    """Luật trận đấu; TeamVecEnv làm trọng tài."""

    match_s: float = 60.0  # một trận; bàn thắng không kết thúc trận
    board_gap: float = 0.5  # m: thành sân lùi ra sau vạch để bóng ra khỏi sân được (không kẹt vào thành)
    touch_radius: float = 0.12  # m: tâm thân vịt gần tâm bóng (bóng đang lăn) hơn mức này coi như chạm bóng
    kickoff_s: float = 2.0  # thời gian bảo vệ quả giao bóng (hết sớm khi bóng rời giữa sân kickoff_moved)
    kickoff_moved: float = 0.1  # m
    kickoff_clear: float = 0.4  # m: đội không giao bóng phải đứng xa bóng bấy nhiêu
    encroach: float = -0.5  # phạt mỗi bước cho vịt lấn vào vùng cấm khi đội kia giao bóng
    throw_flight_s: float = 0.8  # thời gian bay của quả ném vào sân
    throw_inset: float = 0.3  # m: điểm rơi cách mọi vạch ít nhất bấy nhiêu
    gravity: float = 9.81
    return_s: float = 5.0  # sau bàn thắng: thời gian tối đa để các vịt tự chạy về vị trí
    return_tol: float = 0.12  # m: tới vị trí trong khoảng này (và đúng hướng ±0,3 rad) là xong
    return_gain: float = 2.0  # lệnh vận tốc = gain x độ lệch (m), chặn theo giới hạn của tầng đi bộ
    getup_timeout_s: float = 6.0
    stall_speed: float = 0.05  # m/s: dưới mức này bóng coi như đứng yên
    pass_window_s: float = 3.0
    kick_min_speed: float = 0.4  # m/s: cú sút trúng bóng (bóng chạy nhanh hơn mức này khi sút xong)
    shot_min_speed: float = 0.5  # m/s


@configclass
class FieldStatsCfg:
    """Ngưỡng của các số đo bóng kẹt / bu cụm (chỉ để log, không vào phần thưởng)."""

    corner_margin: float = 0.3  # m tính từ hai biên: vùng góc sân
    stalled_speed: float = 0.05  # m/s: bóng coi như đứng yên
    crowd_radius: float = 0.3  # m quanh bóng


def team_size() -> int:
    n = int(os.environ.get("SOCCER_TEAM_SIZE", "2"))
    assert 1 <= n <= MAX_TEAM, n
    return n


def team_field(n: int) -> tuple[float, float, float]:
    """(nửa dài, nửa rộng, nửa bề rộng khung thành): 3 x 2 m cho đội ≤ 2 vịt, 4 x 2,6 m cho đội lớn hơn."""
    return (1.5, 1.0, 0.3) if n <= 2 else (2.0, 1.3, 0.4)


def wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# --- cấu hình môi trường ---------------------------------------------------------------------


@configclass
class TeamSceneCfg(InteractiveSceneCfg):
    terrain = LOW_LEVEL.scene.terrain
    sky_light = LOW_LEVEL.scene.sky_light


@configclass
class TeamObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Chỉ để môi trường có nhóm "policy"; quan sát của từng vịt do TeamVecEnv tính.
        ball = ObsTerm(func=soccer_rl.ball_on_field)

    policy: PolicyCfg = PolicyCfg()


@configclass
class TeamCommandsCfg:
    # Tầng đi bộ đọc tư thế đầu/thân mong muốn; dùng chung cho mọi vịt trong trận.
    head_pose = LOW_LEVEL.commands.head_pose
    body_pose = LOW_LEVEL.commands.body_pose


@configclass
class TeamRewardsCfg:
    pass  # phần thưởng của từng vịt do TeamVecEnv tính


@configclass
class TeamTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TeamActionsCfg:
    pass  # joint_pos_0 ... joint_pos_{2N-1}, thêm trong __post_init__


@configclass
class TeamEventsCfg:
    pass  # đặt lại vị trí từng vịt và bóng, thêm trong __post_init__


@configclass
class MicroDuckSoccerTeamEnvCfg(ManagerBasedRLEnvCfg):
    scene: TeamSceneCfg = TeamSceneCfg(num_envs=512, env_spacing=5.0)
    observations: TeamObservationsCfg = TeamObservationsCfg()
    actions: TeamActionsCfg = TeamActionsCfg()
    commands: TeamCommandsCfg = TeamCommandsCfg()
    rewards: TeamRewardsCfg = TeamRewardsCfg()
    terminations: TeamTerminationsCfg = TeamTerminationsCfg()
    events: TeamEventsCfg = TeamEventsCfg()
    sim = copy.deepcopy(LOW_LEVEL.sim)

    def __post_init__(self):
        n = team_size()
        half_x, half_y, goal_w = team_field(n)
        self.decimation = LOW_LEVEL.decimation * 5  # tầng chiến thuật 10 Hz, tầng kỹ năng 50 Hz
        rules = TeamRulesCfg()
        self.episode_length_s = rules.match_s
        self.sim.dt = LOW_LEVEL.sim.dt
        self.sim.render_interval = LOW_LEVEL.decimation
        self.sim.use_newton_actuators = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # Thân va chạm toàn phần (cần để ngã và đứng dậy thật): nhiều điểm tiếp xúc hơn hẳn bản chỉ có bàn chân.
        self.sim.physics.solver_cfg.nconmax = 64 + 48 * n
        self.sim.physics.solver_cfg.njmax = 8 * self.sim.physics.solver_cfg.nconmax
        self.scene.env_spacing = 2 * (half_x + rules.board_gap) + 1.0
        visuals = bool(os.environ.get("SOCCER_FIELD_VISUALS"))  # cỏ và vạch vôi chỉ cần khi xem
        for name, cfg in field_layout("{ENV_REGEX_NS}/Field", visuals, half_x, half_y, goal_w, rules.board_gap):
            setattr(self.scene, name, cfg)
        self.scene.ball = ball_cfg(BALL_R, BALL_MASS)
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("ball")},
        )
        robot_cfg = MICRODUCK_ALLCOLLISIONS_CFG
        for i in range(2 * n):
            fx, fy = FORMATIONS[n][i % n]
            x, y, rot = fx * half_x, fy * half_y, (0.0, 0.0, 0.0, 1.0)
            if i >= n:  # đội B: đối xứng qua tâm sân, quay mặt về -x
                x, y, rot = -x, -y, (0.0, 0.0, 1.0, 0.0)
            robot = f"robot_{i}"
            setattr(
                self.scene,
                robot,
                robot_cfg.replace(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot_{i}",
                    init_state=robot_cfg.init_state.replace(pos=(x, y, robot_cfg.init_state.pos[2]), rot=rot),
                ),
            )
            setattr(
                self.actions,
                f"joint_pos_{i}",
                soccer_rl.HighLevelVelocityActionCfg(
                    asset_name=robot,
                    term_name=f"joint_pos_{i}",
                    policy_path=VELSTAND,
                    low_level_decimation=LOW_LEVEL.decimation,
                    low_level_actions=copy.deepcopy(LOW_LEVEL.actions.joint_pos),
                    low_level_observations=copy.deepcopy(LOW_LEVEL.observations.policy),
                    getup=True,
                    kick_policy_paths=KICKS,
                    debug_vis=False,
                ),
            )
            setattr(
                self.events,
                f"reset_{robot}",
                EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.5, 0.5)},
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg(robot),
                    },
                ),
            )
            setattr(
                self.events,
                f"reset_{robot}_joints",
                EventTerm(
                    func=mdp.reset_joints_by_offset,
                    mode="reset",
                    params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg(robot)},
                ),
            )
        # Bàn thắng, bóng ra khỏi sân, chạy về vị trí do TeamVecEnv xử lý; trận chỉ kết thúc khi hết giờ.


# --- nhiều người chơi cho rsl_rl / mappo -----------------------------------------------------


class TeamVecEnv(VecEnv):
    """Mỗi trận M có 2N vịt → M*2N người chơi, mỗi người 4 hành động (vx, vy, wz, sút).

    Quan sát của một vịt (trong khung đội mình, rồi xoay theo hướng thân): vị trí, hướng, vận tốc của mình; bóng;
    hai khung thành; N-1 đồng đội và N đối thủ gần nhất; trạng thái trận (đội mình / đội kia giao bóng, thời gian bảo
    vệ còn lại, thời gian trận còn lại, đang chạy về vị trí, mình đang ngã, mình đang sút); hành động trước.
    Phần thưởng (TeamRewardCfg): ghi bàn +10, thủng lưới -3; chuyền thành công, sút trúng hướng khung thành; nằm quá
    lâu không đứng dậy được; bóng lăn về khung thành đối phương (cả đội, thêm cho vịt chạm cuối); vai trò theo tình
    huống (người lấy bóng / bọc lót); đồng đội đứng sát nhau; bóng đứng yên; lấn vùng cấm khi đội kia giao bóng.
    Trong lúc chạy về vị trí sau bàn thắng, hành động của tầng chiến thuật bị bỏ qua và không có phần thưởng dày.
    """

    # Tổng cộng dồn từ lúc tạo (để đánh giá): số bước trận, bóng ở góc, bóng đứng yên, vịt quanh bóng, bước có ≥3 vịt
    # quanh bóng, rồi số lần: giao bóng sau bàn thắng, ném bóng vào sân, sút, sút trúng hướng khung thành, chuyền thành
    # công, ngã, nằm quá lâu phải dựng lại, vịt chưa kịp chạy về vị trí phải dịch chuyển.
    STAT_NAMES = (
        "steps", "corner", "stalled", "crowd", "crowd3",
        "kickoff", "throw_in", "kick", "shot", "pass", "fall", "unrecovered", "return_teleport",
    )
    EVENTS = STAT_NAMES[5:]

    def __init__(
        self,
        env,
        n: int,
        rewards: TeamRewardCfg | None = None,
        rules: TeamRulesCfg | None = None,
        stats: FieldStatsCfg | None = None,
    ):
        self.env = env
        self.rew_cfg = rewards or TeamRewardCfg()
        self.rules = rules or TeamRulesCfg()
        self.stats_cfg = stats or FieldStatsCfg()
        self.n, self.k = n, 2 * n
        self.matches = env.num_envs
        self.num_envs = self.matches * self.k
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg
        self.half_x, self.half_y, self.goal_w = team_field(n)
        self.robots = [env.scene[f"robot_{i}"] for i in range(self.k)]
        self.terms = [env.action_manager.get_term(f"joint_pos_{i}") for i in range(self.k)]
        self.num_actions = self.terms[0].action_dim
        self.ball = env.scene["ball"]
        self.sign = torch.tensor([1.0] * n + [-1.0] * n, device=self.device)  # đội B lật sân
        idx = torch.arange(self.k, device=self.device)
        team = idx // n
        others = idx.unsqueeze(0).expand(self.k, self.k)
        same = team.unsqueeze(1) == team.unsqueeze(0)
        self.mate_idx = others[same & (others != idx.unsqueeze(1))].view(self.k, n - 1)
        self.opp_idx = others[~same].view(self.k, n)
        self.duck_team = team  # (2N,) 0 = đội A, 1 = đội B
        m = self.matches
        self.last_actions = torch.zeros(m, self.k, self.num_actions, device=self.device)
        self.goals = torch.zeros(2, device=self.device)
        self.totals = torch.zeros(len(self.STAT_NAMES), device=self.device)
        self.match_score = torch.zeros(m, 2, device=self.device)  # bàn A, B của trận đang đá
        self.final_score = torch.zeros(m, 2, device=self.device)  # tỷ số trận vừa hết giờ (khi done)
        # Trạng thái trọng tài mỗi trận.
        self.clock = torch.zeros(m, device=self.device)
        self.restart_team = torch.full((m,), -1, dtype=torch.long, device=self.device)  # đội giao bóng (-1: đá tự do)
        self.restart_left = torch.zeros(m, device=self.device)
        self.returning = torch.zeros(m, dtype=torch.bool, device=self.device)  # các vịt đang chạy về vị trí
        self.return_left = torch.zeros(m, device=self.device)
        self.kickoff_team = torch.zeros(m, dtype=torch.long, device=self.device)  # đội giao bóng sau khi chạy về
        self.in_flight = torch.zeros(m, dtype=torch.bool, device=self.device)  # bóng đang được ném vào sân
        self.last_touch = torch.full((m,), -1, dtype=torch.long, device=self.device)  # đội chạm cuối
        self.last_toucher = torch.full((m,), -1, dtype=torch.long, device=self.device)  # vịt chạm cuối
        self.pass_from = torch.full((m,), -1, dtype=torch.long, device=self.device)  # vịt vừa sút trúng bóng
        self.pass_time = torch.zeros(m, device=self.device)
        self.prev_kicking = torch.zeros(m, self.k, dtype=torch.bool, device=self.device)
        self.env.reset()
        self._new_matches(torch.arange(m, device=self.device))
        self._obs = self._observations()

    # --- giao diện VecEnv ---

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf.repeat_interleave(self.k)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf[:] = value.view(self.matches, self.k)[:, 0]

    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self):
        self.env.reset()
        self.last_actions.zero_()
        self.match_score.zero_()
        self._new_matches(torch.arange(self.matches, device=self.device))
        self._obs = self._observations()
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        a = actions.view(self.matches, self.k, self.num_actions).clamp(-1.0, 1.0)
        self.last_actions[:] = a
        _, _, terminated, truncated, extras = self.env.step(a.reshape(self.matches, -1))
        dones = terminated | truncated
        self.clock += self.env.step_dt
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids):
            self._new_matches(done_ids)  # env đã xếp đội hình và đặt bóng giữa sân
            self.final_score[done_ids] = self.match_score[done_ids]
            self.match_score[done_ids] = 0.0
        ev = self._referee(~dones)
        goal_a, goal_b = ev["goal_a"].float(), ev["goal_b"].float()
        self.match_score += torch.stack((goal_a, goal_b), dim=-1)
        team_a = (self.sign > 0).unsqueeze(0)
        scored = torch.where(team_a, goal_a.unsqueeze(1), goal_b.unsqueeze(1))
        conceded = torch.where(team_a, goal_b.unsqueeze(1), goal_a.unsqueeze(1))
        w = self.rew_cfg
        rew = (
            w.goal * scored
            + w.conceded * conceded
            + w.pass_completed * ev["pass"].float()
            + w.shot_on_target * ev["shot"].float()
            + w.kick_attempt * ev["kicked"].float()
            + w.unrecovered_fall * ev["unrecovered"].float()
            + self.rules.encroach * ev["encroach"].float()
        )

        pos, _, _, ball, ball_vel = self._team_frame()
        dense, dist = self._shaping(pos, ball, ball_vel)
        quiet = (dones | self.returning).unsqueeze(1)  # sau reset hoặc đang chạy về vị trí
        rew = rew + torch.where(quiet, torch.zeros_like(dense), dense)

        self.goals += torch.stack((goal_a.sum(), goal_b.sum()))
        step_stats = self._field_stats(dist, ball[:, 0], ball_vel[:, 0], ~dones & ~self.returning)
        self.totals += torch.cat((step_stats, ev["counts"]))
        self.last_actions[dones] = 0.0
        self._obs = self._observations()
        log = dict(extras.get("log", {}))
        live = step_stats[0].clamp_min(1.0)
        log["Soccer/ball_corner_frac"] = (step_stats[1] / live).item()
        log["Soccer/ball_stalled_frac"] = (step_stats[2] / live).item()
        log["Soccer/ducks_near_ball"] = (step_stats[3] / live).item()
        per_min = 60.0 / (self.matches * self.env.step_dt)
        log["Soccer/goals_per_min"] = ((goal_a.sum() + goal_b.sum()) * per_min).item()
        for name, count in zip(self.EVENTS, ev["counts"]):
            log[f"Soccer/{name}_per_min"] = (count * per_min).item()
        out = {"log": log, "time_outs": truncated.repeat_interleave(self.k)}
        return self._obs, rew.view(-1), dones.repeat_interleave(self.k).long(), out

    def close(self):
        self.env.close()

    def _shaping(self, pos, ball, ball_vel):
        """Phần thưởng dày (M, 2N) trong khung đội của từng vịt, và khoảng cách vịt–bóng (M, 2N)."""
        w = self.rew_cfg
        to_goal = torch.tensor((self.half_x, 0.0), device=self.device) - ball
        toward = (ball_vel * to_goal / torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1e-6)).sum(-1)
        toward = toward.clamp(-1.0, 1.0) * (~self.in_flight).float()[:, None]  # quả ném của trọng tài không tính
        toucher = torch.zeros_like(toward)
        has = self.last_toucher >= 0
        toucher[has.nonzero(as_tuple=False).squeeze(-1), self.last_toucher[has]] = 1.0
        dist = torch.linalg.norm(ball - pos, dim=-1)
        striker = torch.zeros_like(dist)
        for t in range(2):
            team_slice = slice(t * self.n, (t + 1) * self.n)
            best = dist[:, team_slice].argmin(-1)
            striker[torch.arange(self.matches, device=self.device), t * self.n + best] = 1.0
        support = 1.0 - striker
        # Chỗ bọc lót: trên đường từ bóng về khung thành nhà (-x trong khung đội), cách bóng support_dist.
        to_home = torch.tensor((-self.half_x, 0.0), device=self.device) - ball
        cover = ball + to_home / torch.linalg.norm(to_home, dim=-1, keepdim=True).clamp_min(1e-6) * w.support_dist
        cover_err = torch.linalg.norm(cover - pos, dim=-1)
        mates = pos[:, self.mate_idx]  # (M, 2N, N-1, 2)
        mate_close = (torch.linalg.norm(mates - pos[:, :, None], dim=-1) < w.mate_dist).float().sum(-1)
        free = ((self.restart_team < 0) & ~self.in_flight)[:, None]
        stalled = free & (torch.linalg.norm(ball_vel, dim=-1) < self.rules.stall_speed)
        return (
            w.ball_to_goal * toward
            + w.ball_to_goal_toucher * toucher * toward
            + w.near_ball * striker * torch.exp(-dist / w.near_ball_std)
            + w.support_pos * support * torch.exp(-cover_err / w.support_std)
            + w.support_near_ball * support * (dist < self.stats_cfg.crowd_radius).float()
            + w.mate_close * mate_close
            + w.stalled * stalled.float()
        ), dist

    # --- trọng tài ---

    def _new_matches(self, ids):
        """Trận mới (sau reset của env): một đội ngẫu nhiên giao bóng."""
        self._start_kickoff(ids, torch.randint(0, 2, (len(ids),), device=self.device))
        self.returning[ids] = False
        self.in_flight[ids] = False
        self.clock[ids] = 0.0
        self.prev_kicking[ids] = False
        for term in self.terms:
            term.override_mask[ids] = False

    def _start_kickoff(self, ids, team):
        self.restart_team[ids] = team
        self.restart_left[ids] = self.rules.kickoff_s
        self.last_touch[ids] = -1
        self.last_toucher[ids] = -1
        self.pass_from[ids] = -1

    def _referee(self, live):
        """Một bước trọng tài. live (M,): trận không vừa reset. Trả về dict: goal_a, goal_b (M,), encroach, pass, shot,
        kicked, unrecovered (M, 2N), counts (len(EVENTS),)."""
        r = self.rules
        dt = self.env.step_dt
        dev = self.device
        counts = torch.zeros(len(self.EVENTS), device=dev)
        ev_idx = {name: i for i, name in enumerate(self.EVENTS)}
        origins = self.env.scene.env_origins[:, :2]
        ball = self.ball.data.root_pos_w.torch[:, :2] - origins
        ball_z = self.ball.data.root_pos_w.torch[:, 2] - self.env.scene.env_origins[:, 2]
        vel = self.ball.data.root_lin_vel_w.torch[:, :2]
        speed = torch.linalg.norm(vel, dim=-1)
        ducks = torch.stack([rb.data.root_pos_w.torch[:, :2] for rb in self.robots], dim=1) - origins[:, None]
        dist = torch.linalg.norm(ducks - ball[:, None], dim=-1)
        play = live & ~self.returning

        # Kỹ năng: ngã, sút (sự kiện cộng dồn trong khối hành động).
        modes = torch.stack([t.mode for t in self.terms], dim=1)
        fell = torch.stack([t.fell for t in self.terms], dim=1) & live[:, None]
        kicked = torch.stack([t.kick_started for t in self.terms], dim=1) & live[:, None]
        for t in self.terms:
            t.fell[:] = False
            t.kick_started[:] = False
        counts[ev_idx["fall"]] = fell.sum()
        counts[ev_idx["kick"]] = kicked.sum()
        down_time = torch.stack(
            [torch.where(t.mode == soccer_rl.HighLevelVelocityAction.DOWN, t.mode_time, 0.0) for t in self.terms], dim=1
        )
        unrecovered = live[:, None] & (down_time > r.getup_timeout_s)
        for i in range(self.k):
            ids = unrecovered[:, i].nonzero(as_tuple=False).squeeze(-1)
            if len(ids):
                self._respawn(i, ids)
        counts[ev_idx["unrecovered"]] = unrecovered.sum()

        # Bóng bay (quả ném) chạm đất trong sân thì hết bay.
        inside = (ball[:, 0].abs() < self.half_x) & (ball[:, 1].abs() < self.half_y)
        self.in_flight &= ~(inside & (ball_z < BALL_R + 0.02))

        # Chạm bóng (bóng đang lăn trên sân), chuyền.
        dmin, who = dist.min(-1)
        touched = play & ~self.in_flight & (dmin < r.touch_radius)
        new_touch = touched & (who != self.last_toucher)
        pf = self.pass_from.clamp(min=0)
        is_pass = (
            new_touch
            & (self.pass_from >= 0)
            & (self.duck_team[who] == self.duck_team[pf])
            & (self.clock - self.pass_time <= r.pass_window_s)
        )
        passes = torch.zeros(self.matches, self.k, dtype=torch.bool, device=dev)
        passes[is_pass.nonzero(as_tuple=False).squeeze(-1), pf[is_pass]] = True
        counts[ev_idx["pass"]] = is_pass.sum()
        self.pass_from = torch.where(new_touch, -1, self.pass_from)
        self.last_touch = torch.where(touched, self.duck_team[who], self.last_touch)
        self.last_toucher = torch.where(touched, who, self.last_toucher)

        # Sút xong: trúng bóng nếu vịt đó chạm bóng cuối và bóng chạy đủ nhanh; sút trúng hướng khung thành đối phương.
        kicking = modes == soccer_rl.HighLevelVelocityAction.KICK
        ended = self.prev_kicking & ~kicking & play[:, None]
        self.prev_kicking = kicking
        own = self.last_toucher[:, None] == torch.arange(self.k, device=dev)[None, :]
        hit = ended & own & (speed[:, None] > r.kick_min_speed)
        hit_match = hit.any(-1)
        self.pass_from = torch.where(hit_match, hit.float().argmax(-1), self.pass_from)
        self.pass_time = torch.where(hit_match, self.clock, self.pass_time)
        ball_t = ball[:, None, :] * self.sign[None, :, None]  # khung đội của từng vịt
        vel_t = vel[:, None, :] * self.sign[None, :, None]
        reach = (self.half_x - ball_t[..., 0]) / vel_t[..., 0].clamp_min(1e-3)
        y_at = ball_t[..., 1] + vel_t[..., 1] * reach
        shots = hit & (vel_t[..., 0] > 0) & (y_at.abs() < self.goal_w) & (speed[:, None] > r.shot_min_speed)
        counts[ev_idx["shot"]] = shots.sum()

        # Quả giao bóng đang được bảo vệ: hết giờ hoặc bóng rời giữa sân thì đá tự do; đội kia lấn vùng cấm bị phạt.
        protected = play & (self.restart_team >= 0)
        moved = torch.linalg.norm(ball, dim=-1) > r.kickoff_moved
        self.restart_left = torch.where(protected, self.restart_left - dt, self.restart_left)
        ended_ko = protected & ((self.restart_left <= 0) | moved)
        self.restart_team = torch.where(ended_ko, -1, self.restart_team)
        protected = protected & ~ended_ko
        encroach = (
            protected[:, None] & (self.duck_team[None, :] != self.restart_team[:, None]) & (dist < r.kickoff_clear)
        )

        # Bàn thắng → bóng về giữa sân, các vịt chạy về vị trí.
        in_mouth = ball[:, 1].abs() < self.goal_w
        goal_a = play & in_mouth & (ball[:, 0] > self.half_x + 0.03)
        goal_b = play & in_mouth & (ball[:, 0] < -self.half_x - 0.03)
        ids = (goal_a | goal_b).nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self._place_ball(ids)
            self.returning[ids] = True
            self.return_left[ids] = r.return_s
            self.kickoff_team[ids] = goal_a[ids].long()  # A ghi bàn → B (1) giao bóng
            self.restart_team[ids] = -1
            self.in_flight[ids] = False
            self.pass_from[ids] = -1

        # Bóng ra khỏi sân (không vào khung thành) → người vô hình ném bóng bay vào sân.
        out = (ball[:, 1].abs() > self.half_y + BALL_R) | ((ball[:, 0].abs() > self.half_x + BALL_R) & ~in_mouth)
        throw = play & ~goal_a & ~goal_b & out & ~self.in_flight
        ids = throw.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            self._throw(ids, ball[ids])
            self.in_flight[ids] = True
            self.pass_from[ids] = -1
        counts[ev_idx["throw_in"]] = throw.sum()

        # Chạy về vị trí.
        counts[ev_idx["return_teleport"]] = self._run_back(live)
        counts[ev_idx["kickoff"]] = self._kickoffs
        return {"goal_a": goal_a, "goal_b": goal_b, "encroach": encroach, "pass": passes, "shot": shots,
                "kicked": kicked, "unrecovered": unrecovered, "counts": counts}

    def _run_back(self, live) -> float:
        """Lệnh chạy về vị trí đội hình cho các trận đang 'returning'. Hết giờ hoặc mọi vịt đã tới: vịt chưa tới bị
        dịch chuyển, bóng đặt lại giữa sân, đội thủng lưới giao bóng. Trả về số vịt bị dịch chuyển."""
        r = self.rules
        self._kickoffs = 0.0
        active = self.returning & live
        self.return_left = torch.where(active, self.return_left - self.env.step_dt, self.return_left)
        if not bool(active.any()):
            for term in self.terms:
                term.override_mask[:] = False
            return 0.0
        origins = self.env.scene.env_origins[:, :2]
        arrived = torch.ones(self.matches, self.k, dtype=torch.bool, device=self.device)
        for i, (robot, term) in enumerate(zip(self.robots, self.terms)):
            target = robot.data.default_root_pose.torch[:, :2]
            target_yaw = 0.0 if i < self.n else math.pi
            pos = robot.data.root_pos_w.torch[:, :2] - origins
            yaw = robot.data.heading_w.torch
            d_w = target - pos
            d = torch.linalg.norm(d_w, dim=-1)
            cos, sin = torch.cos(yaw), torch.sin(yaw)
            d_b = torch.stack((cos * d_w[:, 0] + sin * d_w[:, 1], -sin * d_w[:, 0] + cos * d_w[:, 1]), dim=-1)
            face = torch.where(d > 0.25, torch.atan2(d_w[:, 1], d_w[:, 0]), torch.full_like(d, target_yaw))
            ok = (d < r.return_tol) & (wrap(target_yaw - yaw).abs() < 0.3)
            cmd = torch.stack(
                (
                    (r.return_gain * d_b[:, 0]).clamp(-0.4, 0.4),
                    (r.return_gain * d_b[:, 1]).clamp(-0.3, 0.3),
                    (2.0 * wrap(face - yaw)).clamp(-1.0, 1.0),
                ),
                dim=-1,
            )
            term.override_cmd[:] = torch.where(ok[:, None], torch.zeros_like(cmd), cmd)
            term.override_mask[:] = active
            arrived[:, i] = ok
        finished = active & (arrived.all(-1) | (self.return_left <= 0))
        teleported = 0.0
        ids = finished.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            for i, term in enumerate(self.terms):
                late = ids[~arrived[ids, i]]
                if len(late):
                    self._respawn(i, late)
                    teleported += len(late)
                term.override_mask[ids] = False
            self._place_ball(ids)
            self.returning[ids] = False
            self._start_kickoff(ids, self.kickoff_team[ids])
            self._kickoffs = float(len(ids))
        return teleported

    def _throw(self, ids, ball):
        """Người vô hình ném bóng bay từ chỗ nó đang nằm tới một điểm ngẫu nhiên trong sân (bay throw_flight_s)."""
        r = self.rules
        lim = torch.tensor((self.half_x - r.throw_inset, self.half_y - r.throw_inset), device=self.device)
        target = (torch.rand(len(ids), 2, device=self.device) * 2.0 - 1.0) * lim
        v = torch.zeros(len(ids), 6, device=self.device)
        v[:, :2] = (target - ball) / r.throw_flight_s
        v[:, 2] = r.gravity * r.throw_flight_s / 2.0
        self.ball.write_root_velocity_to_sim_index(root_velocity=v, env_ids=ids)

    def _place_ball(self, ids):
        """Bóng về giữa sân (sau bàn thắng)."""
        pose = torch.zeros(len(ids), 7, device=self.device)
        pose[:, :2] = self.env.scene.env_origins[ids, :2]
        pose[:, 2] = self.env.scene.env_origins[ids, 2] + BALL_R + 0.005
        pose[:, 6] = 1.0
        self.ball.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _respawn(self, i, ids):
        """Dự phòng: vịt i của các trận ids về vị trí đội hình, đứng thẳng, khớp và kỹ năng về mặc định."""
        robot = self.robots[i]
        pose = robot.data.default_root_pose.torch[ids].clone()
        pose[:, :3] += self.env.scene.env_origins[ids]
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.torch[ids].clone(),
            velocity=torch.zeros_like(robot.data.default_joint_vel.torch[ids]),
            env_ids=ids,
        )
        term = self.terms[i]
        term.low_level_actions[ids] = 0.0
        term.mode[ids] = term.WALK
        term.mode_time[ids] = 0.0
        term.up_time[ids] = 0.0

    def _field_stats(self, dist, ball, ball_vel, live) -> torch.Tensor:
        """Tổng trên các trận đang đá (live) của: 1, bóng ở góc, bóng đứng yên, số vịt quanh bóng, có ≥3 vịt quanh bóng.

        dist (M, 2N) khoảng cách vịt–bóng; ball, ball_vel (M, 2) trong khung đội A (góc và tốc độ không đổi khi lật).
        """
        c = self.stats_cfg
        corner = (ball[:, 0].abs() > self.half_x - c.corner_margin) & (ball[:, 1].abs() > self.half_y - c.corner_margin)
        stalled = torch.linalg.norm(ball_vel, dim=-1) < c.stalled_speed
        crowd = (dist < c.crowd_radius).sum(-1).float()
        live = live.float()
        return torch.stack(
            (live.sum(), (corner * live).sum(), (stalled * live).sum(), (crowd * live).sum(), ((crowd >= 3) * live).sum())
        )

    # --- quan sát ---

    def _team_frame(self):
        """Vị trí, hướng, vận tốc thân của 2N vịt và bóng, trong khung đội của từng vịt (đội B lật 180°)."""
        origins = self.env.scene.env_origins[:, None, :2]
        s = self.sign[None, :, None]
        pos = torch.stack([r.data.root_pos_w.torch[:, :2] for r in self.robots], dim=1) - origins
        yaw = torch.stack([r.data.heading_w.torch for r in self.robots], dim=1)
        vel_b = torch.stack([r.data.root_lin_vel_b.torch[:, :2] for r in self.robots], dim=1)
        ball = (self.ball.data.root_pos_w.torch[:, None, :2] - origins) * s
        ball_vel = self.ball.data.root_lin_vel_w.torch[:, None, :2] * s
        yaw_t = yaw + (self.sign < 0).float()[None, :] * math.pi
        return pos * s, yaw_t, vel_b, ball, ball_vel

    @staticmethod
    def _body(vec, yaw):
        """Xoay vec (..., 2) vào khung thân có góc yaw (khớp các chiều đầu của vec)."""
        while yaw.dim() < vec.dim() - 1:
            yaw = yaw.unsqueeze(-1)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        return torch.stack((cos * vec[..., 0] + sin * vec[..., 1], -sin * vec[..., 0] + cos * vec[..., 1]), dim=-1)

    def _nearest(self, rel, idx):
        """rel (M, k, k, 2) vị trí các vịt nhìn từ từng vịt; idx (k, j) → j vịt theo idx, sắp theo khoảng cách."""
        picked = rel[:, torch.arange(self.k, device=self.device).unsqueeze(1), idx]  # (M, k, j, 2)
        order = torch.linalg.norm(picked, dim=-1).argsort(dim=-1)
        return torch.gather(picked, 2, order.unsqueeze(-1).expand_as(picked))

    def _observations(self) -> TensorDict:
        pos, yaw, vel_b, ball, ball_vel = self._team_frame()
        # Vị trí mọi vịt trong khung đội của vịt quan sát: lật theo dấu của vịt quan sát.
        raw = torch.stack([r.data.root_pos_w.torch[:, :2] for r in self.robots], dim=1) - self.env.scene.env_origins[:, None, :2]
        seen = raw[:, None, :, :] * self.sign[None, :, None, None]  # (M, quan sát, bị nhìn, 2)
        rel = self._body(seen - pos[:, :, None, :], yaw)
        goal_opp = torch.tensor((self.half_x, 0.0), device=self.device)
        ours = self.restart_team[:, None] == self.duck_team[None, :]
        theirs = (self.restart_team[:, None] >= 0) & ~ours
        restart_left = (self.restart_left / self.rules.kickoff_s).clamp(0.0, 1.0)[:, None].expand(-1, self.k)
        time_left = (1.0 - self.env.episode_length_buf.float() / self.max_episode_length)[:, None].expand(-1, self.k)
        returning = self.returning.float()[:, None].expand(-1, self.k)
        modes = torch.stack([t.mode for t in self.terms], dim=1)
        down = (modes == soccer_rl.HighLevelVelocityAction.DOWN).float()
        kicking = (modes == soccer_rl.HighLevelVelocityAction.KICK).float()
        game = torch.stack((ours.float(), theirs.float(), restart_left, time_left, returning, down, kicking), dim=-1)
        policy = torch.cat(
            (
                pos,
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
                vel_b,
                self._body(ball - pos, yaw),
                self._body(ball_vel, yaw),
                self._body(goal_opp - pos, yaw),
                self._body(-goal_opp - pos, yaw),
                self._nearest(rel, self.mate_idx).flatten(2),
                self._nearest(rel, self.opp_idx).flatten(2),
                game,
                self.last_actions,
            ),
            dim=-1,
        )
        critic = torch.cat((policy, ball, ball_vel), dim=-1)
        return TensorDict(
            {"policy": policy.flatten(0, 1), "critic": critic.flatten(0, 1)}, batch_size=[self.num_envs]
        )
