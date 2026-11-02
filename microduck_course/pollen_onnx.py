"""Đổi não ONNX của Pollen Robotics (huggingface.co/pollen-robotics/microduck-policies, Apache-2.0) sang TorchScript.

  python pollen_onnx.py <thư mục .onnx> <thư mục ra> velstand ball_kick_left ball_kick_right

Mỗi file là một MLP: (obs - mean) / std → Gemm/ELU ×3 → Gemm, vào 61 chiều (giao kèo triển khai của MicroDuck, trùng
nhóm quan sát "policy" của bài Isaac Lab), ra 14 mục tiêu khớp (tỷ lệ 1.0, cộng tư thế gốc). File .pt ra dùng được
như policies/walk_rough.pt. So kết quả với bộ chạy ONNX tham chiếu trên 256 quan sát ngẫu nhiên.
"""

import os
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator


class PollenMLP(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, layers: list[torch.nn.Linear]):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.std
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = torch.nn.functional.elu(x)
        return x


def convert(path: str) -> tuple[PollenMLP, dict]:
    model = onnx.load(path)
    graph = model.graph
    ops = [n.op_type for n in graph.node]
    assert ops == ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"], ops
    init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}
    sub, div = graph.node[0], graph.node[1]
    mean, std = init[sub.input[1]], init[div.input[1]]
    layers = []
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        attrs = {a.name: a for a in node.attribute}
        assert attrs["transB"].i == 1 and ("transA" not in attrs or attrs["transA"].i == 0), node
        w, b = init[node.input[1]], init[node.input[2]]
        layer = torch.nn.Linear(w.shape[1], w.shape[0])
        layer.weight.data.copy_(w)
        layer.bias.data.copy_(b)
        layers.append(layer)
    net = PollenMLP(mean, std, layers).eval()
    meta = {p.key: p.value for p in model.metadata_props}
    # So với bộ chạy ONNX tham chiếu.
    x = np.random.default_rng(0).normal(size=(256, mean.shape[-1])).astype(np.float32)
    ref = np.concatenate([ReferenceEvaluator(model).run(None, {"obs": x[i : i + 1]})[0] for i in range(len(x))])
    with torch.no_grad():
        got = net(torch.from_numpy(x)).numpy()
    err = float(np.abs(ref - got).max())
    assert err < 1e-4, err
    meta["max_abs_err_vs_onnx"] = f"{err:.2e}"
    return net, meta


if __name__ == "__main__":
    src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(dst, exist_ok=True)
    for name in names:
        net, meta = convert(os.path.join(src, f"{name}.onnx"))
        out = os.path.join(dst, f"{name}.pt")
        torch.jit.script(net).save(out)
        print(f"[ONNX] {name}: {out} | sai lệch so với ONNX {meta['max_abs_err_vs_onnx']}")
        for key in ("observation_names", "joint_names", "default_joint_pos", "action_scale"):
            print(f"[ONNX]   {key}: {meta.get(key)}")
