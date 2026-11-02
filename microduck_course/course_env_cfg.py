"""Sân "bàn làm việc tí hon" cho MicroDuck: bài đi bộ của Isaac Lab + vật dụng NVIDIA SimReady (CC-BY-4.0).

Chỉ thêm vật dụng vào cảnh; robot, quan sát, hành động giữ nguyên nên chạy được chính sách MicroDuck đất phẳng
hoặc địa hình. 9 con vịt ở lưới 3x3 (cách 2 m); bàn phím xếp thành lưới caro quanh mỗi ô, nên muốn ra khỏi ô vịt phải
bước qua "bậc" 2 cm. Vật dụng đặt chung cho cả sân (ngoài vùng từng môi trường) và là vật tĩnh: có va chạm, không bị
đẩy đi. Vịt đi bộ "mù" (không camera), nên đâm vào, vấp, trèo lên là chuyện bình thường.
"""

import math
import re

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from ..microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg

STATIC_DIR = "/simready_static"  # ~/Documents/Codex/simready_static; tham chiếu bản gốc ở /simready

KEYBOARD = "warehouse_computer_keyboard_a01"  # 44 x 14 cm, cao 2 cm
BOOK = "book_open_a01"  # 32 x 21 cm, cao 3,7 cm
FIRST_AID = "safety_firstAidKit_case_soft_a01"  # 26 x 21 cm, cao 8 cm
CHOCK = "vehicleSupport_wheel_chock_b01"  # 20 x 12 cm, cao 11 cm, hình nêm


def prop(name, asset, x, y, yaw_deg=0.0):
    """Một vật SimReady tĩnh đặt chung cho cả sân, gốc ở đáy vật, xoay quanh trục z.

    Dùng bản tĩnh do make_static.py tạo (biến thể Newton, đã bỏ vật rắn). Vật rắn tự do dùng chung thêm khớp tự do vào
    mảng khớp chung của Newton và làm lệch chỉ số động cơ của vịt (lỗi "Invalid indexing in slice").
    """
    half = math.radians(yaw_deg) / 2
    return AssetBaseCfg(
        prim_path=f"/World/Course/{name}",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{STATIC_DIR}/{asset}.usda"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.0), rot=(0.0, 0.0, math.sin(half), math.cos(half))),
    )


def desk_layout():
    """Danh sách (tên, vật, x, y, góc) của sân."""
    items = []
    lines = (-1.0, 1.0)  # ranh giới giữa các ô 2 x 2 m
    steps = [-2.75 + 0.5 * i for i in range(12)]
    # Lưới caro bằng bàn phím, chừa chỗ ở giao điểm cho cột.
    for c in lines:
        for t in steps:
            if min(abs(t - line) for line in lines) < 0.3:
                continue
            items.append((f"kb_x{c:+.0f}_{t:+.2f}", KEYBOARD, c, t, 0.0))
            items.append((f"kb_y{c:+.0f}_{t:+.2f}", KEYBOARD, t, c, 90.0))
    # Cột túi cứu thương ở bốn giao điểm.
    for x in lines:
        for y in lines:
            items.append((f"pillar_{x:+.0f}_{y:+.0f}", FIRST_AID, x, y, 45.0))
    # Trong mỗi ô: hai sách mở làm tấm lát, hai chặn bánh xe làm dốc hướng vào giữa ô.
    for cx in (-2.0, 0.0, 2.0):
        for cy in (-2.0, 0.0, 2.0):
            tag = f"{cx:+.0f}_{cy:+.0f}"
            items.append((f"book_a_{tag}", BOOK, cx + 0.7, cy + 0.7, 45.0))
            items.append((f"book_b_{tag}", BOOK, cx - 0.7, cy - 0.7, 45.0))
            items.append((f"chock_a_{tag}", CHOCK, cx - 0.7, cy + 0.7, -45.0))
            items.append((f"chock_b_{tag}", CHOCK, cx + 0.7, cy - 0.7, 135.0))
    # Viền ngoài: túi cứu thương xen chặn bánh xe.
    border = [-3.0 + 0.6 * i for i in range(11)]
    for i, t in enumerate(border):
        asset, yaw = (FIRST_AID, 0.0) if i % 2 == 0 else (CHOCK, 90.0)
        items.append((f"rim_s_{i}", asset, t, -3.4, yaw))
        items.append((f"rim_n_{i}", asset, t, 3.4, yaw + 180.0))
        items.append((f"rim_w_{i}", asset, -3.4, t, yaw + 90.0))
        items.append((f"rim_e_{i}", asset, 3.4, t, yaw - 90.0))
    return items


@configclass
class MicroDuckCourseEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        # Cấu hình đất phẳng chỉ cho MJWarp 16 điểm tiếp xúc mỗi môi trường; vịt chạm vật dụng thì cần nhiều hơn.
        # Nâng như play_mode của bản địa hình.
        self.sim.physics.solver_cfg.nconmax = 200
        self.sim.physics.solver_cfg.njmax = 1024
        self.scene.env_spacing = 2.0
        for name, asset, x, y, yaw in desk_layout():
            name = re.sub(r"[^A-Za-z0-9_]", "_", name.replace("+", "p").replace("-", "m"))  # tên prim USD hợp lệ
            setattr(self.scene, name, prop(name, asset, x, y, yaw))
        # Vịt xuất phát gần giữa ô, không đè lên sách và chặn bánh xe.
        self.events.reset_base.params["pose_range"].update({"x": (-0.35, 0.35), "y": (-0.35, 0.35)})
        self.sim.visualizer_cfgs.eye = (4.5, -4.5, 3.2)
        self.sim.visualizer_cfgs.lookat = (0.0, 0.0, 0.0)
