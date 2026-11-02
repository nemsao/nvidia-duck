"""Bài RL 1: một con MicroDuck học ghi bàn vào khung thành đối phương.

Hai tầng, theo mẫu bài điều hướng của Isaac Lab (contrib/navigation):
  - tầng dưới: chính sách đi bộ MicroDuck đã train (TorchScript, đóng băng) chạy 50 Hz;
  - tầng trên (học ở đây): mỗi 0,1 s ra lệnh (vx, vy, wz) cho tầng dưới.
Mỗi môi trường có một sân riêng (chỉ phần có va chạm khi train), một vịt, một bóng Ø8 cm.
"""

import math
import os

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ..microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from . import soccer_rl
from .soccer_env_cfg import FIELD_X, GOAL_HALF_W, ball_cfg, field_layout

# Chính sách đi bộ xuất từ lượt train Rough-MicroDuck (xem policies/README).
WALK_POLICY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "walk_rough.pt")
GOAL_LINE = FIELD_X + 0.03  # tâm bóng qua vạch vôi thêm 3 cm mới tính là vào
LOW_LEVEL = MicroDuckVelocityFlatEnvCfg()


@configclass
class ActionsCfg:
    # Tên "joint_pos": quan sát và sự kiện encoder của MicroDuck tìm khối này theo tên.
    joint_pos = soccer_rl.HighLevelVelocityActionCfg(
        asset_name="robot",
        policy_path=WALK_POLICY,
        low_level_decimation=LOW_LEVEL.decimation,
        low_level_actions=LOW_LEVEL.actions.joint_pos,
        low_level_observations=LOW_LEVEL.observations.policy,
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot = ObsTerm(func=soccer_rl.robot_on_field)
        robot_vel = ObsTerm(func=soccer_rl.robot_lin_vel_b)
        ball = ObsTerm(func=soccer_rl.ball_in_robot_frame)
        goal = ObsTerm(func=soccer_rl.goal_in_robot_frame, params={"goal_x": FIELD_X})
        own_goal = ObsTerm(func=soccer_rl.goal_in_robot_frame, params={"goal_x": -FIELD_X})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        ball_field = ObsTerm(func=soccer_rl.ball_on_field)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # Ghi bàn là phần thưởng chính; các phần nhỏ chỉ để định hướng lúc đầu.
    scored = RewTerm(func=soccer_rl.ball_crossed, weight=100.0, params={"goal_line_x": GOAL_LINE, "half_width": GOAL_HALF_W})
    own_goal = RewTerm(
        func=soccer_rl.ball_crossed, weight=-50.0, params={"goal_line_x": -GOAL_LINE, "half_width": GOAL_HALF_W}
    )
    ball_to_goal = RewTerm(func=soccer_rl.ball_velocity_to_goal, weight=2.0, params={"goal_x": FIELD_X})
    near_ball = RewTerm(func=soccer_rl.near_ball, weight=0.1, params={"std": 0.3})
    fell = RewTerm(func=mdp.is_terminated_term, weight=-20.0, params={"term_keys": "fell_over"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    scored = DoneTerm(func=soccer_rl.ball_crossed, params={"goal_line_x": GOAL_LINE, "half_width": GOAL_HALF_W})
    own_goal = DoneTerm(func=soccer_rl.ball_crossed, params={"goal_line_x": -GOAL_LINE, "half_width": GOAL_HALF_W})


@configclass
class MicroDuckSoccerScoreEnvCfg(MicroDuckVelocityFlatEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.decimation = LOW_LEVEL.decimation * 5  # tầng trên 10 Hz, tầng dưới 50 Hz
        self.sim.render_interval = LOW_LEVEL.decimation
        self.episode_length_s = 20.0
        self.scene.env_spacing = 4.0  # sân 3,3 x 2,3 m
        self.sim.physics.solver_cfg.nconmax = 64  # chân, bóng, thành sân (cấu hình đi bộ chỉ có 16)
        self.sim.physics.solver_cfg.njmax = 400
        for name, cfg in field_layout(root="{ENV_REGEX_NS}/Field", visuals=False):
            setattr(self.scene, name, cfg)
        self.scene.ball = ball_cfg()
        self.commands.base_velocity.debug_vis = False  # lệnh vận tốc thật đến từ tầng trên
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (-1.2, -0.2), "y": (-0.8, 0.8), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)
        }
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {"x": (-0.6, 0.8), "y": (-0.7, 0.7)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("ball")},
        )

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 4
        for name, cfg in field_layout(root="{ENV_REGEX_NS}/Field", visuals=True):
            setattr(self.scene, name, cfg)
