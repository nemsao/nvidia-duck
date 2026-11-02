"""Tạo bản "vật tĩnh" của vật dụng SimReady cho sân MicroDuck.

Mỗi file /simready_static/<tên>.usda tham chiếu bản gốc (không sửa bản gốc), chọn biến thể vật lý Newton, rồi xoá
PhysicsRigidBodyAPI/PhysicsMassAPI trên các thân: Newton chỉ còn thấy hình va chạm nên coi là vật tĩnh. Bộ nạp USD
của Newton vẫn nạp thân có physics:rigidBodyEnabled = False như vật động, và vật động dùng chung thêm khớp tự do vào
mảng khớp chung, làm lệch chỉ số động cơ của vịt.
Chạy trong container: python make_static.py <tên vật> ...
"""

import os
import sys

from pxr import Sdf, Usd, UsdPhysics

SRC, DST = "/simready", "/simready_static"
VARIANTS = {"Newton": "Enabled", "PhysX": "Disabled", "MuJoCo": "Disabled"}
DROP = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]

os.makedirs(DST, exist_ok=True)
for name in sys.argv[1:]:
    out = f"{DST}/{name}.usda"
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(f"{SRC}/{name}/simready_usd/sm_{name}_01.usd")
    for vset, value in VARIANTS.items():
        if root.GetVariantSets().HasVariantSet(vset):
            root.GetVariantSets().GetVariantSet(vset).SetVariantSelection(value)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for prim in bodies:
        schemas = Sdf.TokenListOp()
        schemas.deletedItems = DROP
        stage.OverridePrim(prim.GetPath()).SetMetadata("apiSchemas", schemas)
    stage.GetRootLayer().Save()
    check = Usd.Stage.Open(out)
    left = [p for p in check.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    every = check.Traverse(Usd.TraverseInstanceProxies())
    colliders = [p for p in every if p.HasAPI(UsdPhysics.CollisionAPI)]
    print(f"{name}: bỏ vật rắn ở {len(bodies)} thân (còn {len(left)}), {len(colliders)} hình va chạm")
