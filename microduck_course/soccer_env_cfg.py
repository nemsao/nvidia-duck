"""Sân bóng tí hon cho MicroDuck: vịt dẫn bóng Ø8 cm vào khung thành (bộ chiến thuật viết tay, để xem).

Bộ não đi bộ là chính sách MicroDuck đã train (đất phẳng hoặc địa hình); lệnh vận tốc đến từ bộ chiến thuật trong
soccer.py thay cho lệnh ngẫu nhiên. Sân 3 x 2 m (cỡ vịt cao ~20 cm), thành sân thấp giữ bóng trong sân, hai khung
thành rộng 0,6 m. Vịt tấn công khung thành cam bên +x; ghi bàn thì bóng và vịt về vị trí mới.
Mặt cỏ, vạch vôi chỉ để nhìn; thành sân, cột dọc, xà, lưới có va chạm (vật tĩnh dùng chung).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab.envs import mdp
from ..microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from .soccer import SoccerVelocityCommandCfg, ball_in_goal

FIELD_X, FIELD_Y = 1.5, 1.0  # nửa chiều dài, nửa chiều rộng
GOAL_HALF_W, GOAL_H, GOAL_DEPTH = 0.3, 0.25, 0.22
BALL_R = 0.04
LINE_W = 0.02
GRASS = (0.13, 0.45, 0.16)
WHITE = (0.95, 0.95, 0.95)
BOARD = (0.85, 0.85, 0.80)
GOAL_COLORS = {1: (1.0, 0.45, 0.05), -1: (0.10, 0.35, 0.95)}  # cam (vịt tấn công), xanh


def box(root, name, center, size, color, collide, yaw_deg=0.0):
    name = name.replace("+", "p").replace("-", "m")  # tên prim USD hợp lệ
    half = math.radians(yaw_deg) / 2
    return name, AssetBaseCfg(
        prim_path=f"{root}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
            collision_props=sim_utils.CollisionPropertiesCfg() if collide else None,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center, rot=(0.0, 0.0, math.sin(half), math.cos(half))),
    )


def field_layout(
    root="/World/Field", visuals=True, half_x=FIELD_X, half_y=FIELD_Y, goal_half_w=GOAL_HALF_W, board_gap=0.15
):
    """Các khối của sân. root="{ENV_REGEX_NS}/Field" cho mỗi môi trường một sân (lúc train);
    visuals=False bỏ mặt cỏ và vạch vôi, chỉ giữ phần có va chạm (nhẹ hơn khi có hàng nghìn sân).
    half_x, half_y, goal_half_w: nửa chiều dài, nửa chiều rộng sân và nửa bề rộng khung thành.
    board_gap: thành sân cách vạch biên bấy nhiêu (m); đủ rộng thì bóng ra biên được và trọng tài cho đá lại."""
    FIELD_X, FIELD_Y, GOAL_HALF_W = half_x, half_y, goal_half_w  # noqa: N806
    items = []

    def box_(name, center, size, color, collide, yaw_deg=0.0):
        if collide or visuals:
            items.append(box(root, name, center, size, color, collide, yaw_deg))

    z_line = 0.0015
    # Mặt cỏ và vạch vôi (chỉ để nhìn, mỏng 1-2 mm).
    margin = max(0.3, board_gap + 0.15)
    box_("grass", (0.0, 0.0, 0.0005), (2 * (FIELD_X + margin), 2 * (FIELD_Y + margin), 0.001), GRASS, False)
    for s in (-1, 1):
        box_(f"touch_{s:+d}", (0.0, s * FIELD_Y, z_line), (2 * FIELD_X, LINE_W, 0.001), WHITE, False)
        box_(f"goalline_{s:+d}", (s * FIELD_X, 0.0, z_line), (LINE_W, 2 * FIELD_Y, 0.001), WHITE, False)
        # Vòng cấm địa 0,35 x 1,0 m.
        box_x = s * (FIELD_X - 0.35)
        box_(f"area_front_{s:+d}", (box_x, 0.0, z_line), (LINE_W, 1.0, 0.001), WHITE, False)
        for t in (-1, 1):
            box_(f"area_side_{s:+d}{t:+d}", (s * (FIELD_X - 0.175), t * 0.5, z_line), (0.35, LINE_W, 0.001), WHITE, False)
    box_("halfway", (0.0, 0.0, z_line), (LINE_W, 2 * FIELD_Y, 0.001), WHITE, False)
    # Vòng tròn giữa sân bán kính 0,3 m ghép từ 32 đoạn.
    r, k = 0.3, 32
    for i in range(k):
        a = 2 * math.pi * (i + 0.5) / k
        box_(f"circle_{i}", (r * math.cos(a), r * math.sin(a), z_line), (2 * math.pi * r / k + 0.004, LINE_W, 0.001), WHITE, False, math.degrees(a) + 90)
    # Thành sân cao 6 cm, chừa miệng khung thành.
    wall_h, wall_t, gap = 0.06, 0.03, board_gap
    for s in (-1, 1):
        box_(f"board_long_{s:+d}", (0.0, s * (FIELD_Y + gap), wall_h / 2), (2 * (FIELD_X + gap) + wall_t, wall_t, wall_h), BOARD, True)
        side_len = FIELD_Y + gap - GOAL_HALF_W
        for t in (-1, 1):
            cy = t * (GOAL_HALF_W + side_len / 2)
            box_(f"board_end_{s:+d}{t:+d}", (s * (FIELD_X + gap), cy, wall_h / 2), (wall_t, side_len, wall_h), BOARD, True)
    # Khung thành: 2 cột dọc, xà ngang, lưới sau và hai bên.
    post = 0.025
    for s in (-1, 1):
        color = GOAL_COLORS[s]
        gx = s * FIELD_X
        for t in (-1, 1):
            box_(f"post_{s:+d}{t:+d}", (gx, t * GOAL_HALF_W, GOAL_H / 2), (post, post, GOAL_H), color, True)
            box_(f"net_side_{s:+d}{t:+d}", (gx + s * GOAL_DEPTH / 2, t * GOAL_HALF_W, GOAL_H / 2), (GOAL_DEPTH, 0.006, GOAL_H), WHITE, True)
        box_(f"bar_{s:+d}", (gx, 0.0, GOAL_H), (post, 2 * GOAL_HALF_W + post, post), color, True)
        box_(f"net_back_{s:+d}", (gx + s * GOAL_DEPTH, 0.0, GOAL_H / 2), (0.006, 2 * GOAL_HALF_W, GOAL_H), WHITE, True)
        box_(f"net_top_{s:+d}", (gx + s * GOAL_DEPTH / 2, 0.0, GOAL_H), (GOAL_DEPTH, 2 * GOAL_HALF_W, 0.004), WHITE, False)
    return items


def ball_cfg(radius=BALL_R, mass=0.06):
    """Bóng Ø8 cm, 60 g (so với vịt cao ~20 cm), là quả cầu toán học, mỗi môi trường một quả.
    Trận đồng đội dùng bóng Ø7 cm, 15 g như bóng Pollen train kỹ năng sút.

    Bóng golf SimReady va chạm bằng lưới tam giác làm MJWarp mất ổn định: vịt ngã sau 1-7 s dù không chạm bóng;
    quả cầu thì vịt đi trọn lượt 40 s.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, radius + 0.005)),
    )


@configclass
class MicroDuckSoccerEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        # Cấu hình đất phẳng chỉ cho MJWarp 16 điểm tiếp xúc mỗi môi trường; nâng như play_mode của bản địa hình.
        self.sim.physics.solver_cfg.nconmax = 200
        self.sim.physics.solver_cfg.njmax = 1024
        self.episode_length_s = 40.0
        for name, cfg in field_layout():
            setattr(self.scene, name, cfg)
        self.scene.ball = ball_cfg()
        # Lệnh vận tốc từ bộ chiến thuật, cùng giới hạn tốc độ như lúc train.
        old = self.commands.base_velocity
        self.commands.base_velocity = SoccerVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1e9, 1e9),
            debug_vis=False,
            ranges=old.ranges,
            goal_xy=(FIELD_X + 0.1, 0.0),
        )
        # Mỗi lượt: bóng ở quanh giữa sân, vịt ở phần sân nhà, quay hướng ngẫu nhiên.
        self.events.reset_base.params["pose_range"] = {
            "x": (-1.1, -0.6), "y": (-0.6, 0.6), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)
        }
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.3, 0.4), "y": (-0.5, 0.5)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("ball"),
            },
        )
        self.terminations.goal = DoneTerm(func=ball_in_goal, params={"goal_line_x": FIELD_X + 0.03, "half_width": GOAL_HALF_W})
        self.sim.visualizer_cfgs.eye = (0.0, -2.6, 1.9)
        self.sim.visualizer_cfgs.lookat = (0.0, 0.0, 0.0)
