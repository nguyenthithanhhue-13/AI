"""CNN phân loại đoạn đường (thay MLP `edge`): ảnh cắt 24x48x3 -> (kiểu nét 5 lớp, bậc thang 2 lớp, một chiều 3 lớp).

Huấn luyện bằng PyTorch (GPU nếu có), xuất ONNX; khi dự đoán chỉ cần onnxruntime.
Trên validation (26.738 mảnh, học train): 1 / 1 / 2 lỗi cho kiểu nét / bậc thang / một chiều (MLP cũ: 1 / 10 / 16).

    python src/cv_edge.py dev        # học train, đo validation   -> outputs/models_dev/edge_cnn.onnx
    python src/cv_edge.py final      # học train + validation     -> outputs/models_final/edge_cnn.onnx
"""
import sys
import time

import numpy as np

from common import *

HEADS = (5, 2, 3)


def prep(x):
    """uint8 (n, 3456) -> float32 (n, 3, 24, 48) trong [-0.5, 0.5]."""
    return x.reshape(-1, 24, 48, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - 0.5


def build_net():
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    return nn.Sequential(block(3, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2), block(64, 128), nn.MaxPool2d(2),
                         block(128, 192), nn.Flatten(), nn.Dropout(0.2), nn.Linear(192 * 3 * 6, 256), nn.ReLU(inplace=True),
                         nn.Linear(256, sum(HEADS)))


def main(mode, epochs=6):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / "cvdata_train.npz"); X, Y = d["edge_x"], d["edge_y"]
    dv = np.load(CACHE / "cvdata_validation.npz"); Xv, Yv = dv["edge_x"], dv["edge_y"]
    if mode == "final":
        X, Y = np.concatenate([X, Xv]), np.concatenate([Y, Yv])
    print(f"CNN đoạn đường ({dev}): {len(X)} mẫu huấn luyện", flush=True)
    net = build_net().to(dev)
    print("tham số:", sum(p.numel() for p in net.parameters()))
    bs = 256
    steps = epochs * (len(X) // bs + 1)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    # GPU 4 GB: giữ dữ liệu ở RAM, chuyển từng lô (để cả mảng trên GPU gây tráo bộ nhớ và chậm hơn nhiều)
    Xg = torch.from_numpy(X); Yg = torch.from_numpy(Y.astype(np.int64))
    Xvg = torch.from_numpy(Xv); Yvg = torch.from_numpy(Yv.astype(np.int64))

    def to_float(xb):
        return xb.view(-1, 24, 48, 3).permute(0, 3, 1, 2).float() / 255.0 - 0.5

    def heads(z):
        out, s = [], 0
        for h in HEADS:
            out.append(z[:, s:s + h]); s += h
        return out

    def evaluate():
        net.eval(); err = [0, 0, 0]
        with torch.no_grad():
            for i in range(0, len(Xvg), 2048):
                z = heads(net(to_float(Xvg[i:i + 2048].to(dev))))
                for k in range(3):
                    y = Yvg[i:i + 2048, k].to(dev); v = y >= 0
                    err[k] += (z[k].argmax(1)[v] != y[v]).sum().item()
        net.train()
        return err

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            if (i // bs) % 200 == 0:
                print(f"      ep {ep + 1} bước {i // bs}/{len(X) // bs} ({time.time() - t0:.0f}s)", flush=True)
            idx = perm[i:i + bs]
            xb = to_float(Xg[idx].to(dev)); yb = Yg[idx].to(dev)
            if np.random.rand() < 0.5:                   # lật dọc: không đổi nhãn
                xb = xb.flip(2)
            if np.random.rand() < 0.5:                   # lật ngang: đổi chiều một chiều (1 <-> 2)
                xb = xb.flip(3)
                ow = yb[:, 2]
                yb[:, 2] = torch.where(ow == 1, 2, torch.where(ow == 2, 1, ow))
            xb = xb * (1 + 0.1 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)) + 0.05 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)
            z = heads(net(xb))
            # đầu ra nào không có nhãn hợp lệ trong lô (vd chữ thời tiết ở lô cuối nhỏ) thì bỏ qua: tránh loss = nan
            loss = sum(F.cross_entropy(z[k], yb[:, k], label_smoothing=0.02, ignore_index=-1)
                       for k in range(3) if (yb[:, k] >= 0).any())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"   epoch {ep + 1}/{epochs} loss {loss.item():.4f} lỗi trên validation (kiểu nét / bậc thang / một chiều) {evaluate()}  ({time.time() - t0:.0f}s)", flush=True)
    folder = OUT / f"models_{mode}"
    folder.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, folder / "edge_cnn.pt")
    export(mode)


def export(mode):
    import torch
    net = build_net()
    net.load_state_dict(torch.load(OUT / f"models_{mode}" / "edge_cnn.pt"))
    net.eval()
    out = OUT / f"models_{mode}" / "edge_cnn.onnx"
    torch.onnx.export(net, torch.zeros(2, 3, 24, 48), str(out), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    (OUT / f"models_{mode}" / "edge_cnn_params.txt").write_text(str(sum(p.numel() for p in net.parameters())))
    print("đã lưu", out)


class EdgeCNN:
    """Cùng giao diện với MLP: predict_proba(X uint8) -> [kiểu nét, bậc thang, một chiều]."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        p = str(path).replace("edge_cnn.onnx", "edge_cnn_params.txt")
        self.n_params = int(open(p).read()) if os.path.exists(p) else 0

    def predict_proba(self, X, batch=4096):
        outs = []
        for i in range(0, len(X), batch):
            z = self.sess.run(None, {"x": prep(np.asarray(X[i:i + batch]))})[0]
            res, s = [], 0
            for h in HEADS:
                zz = z[:, s:s + h]; zz = zz - zz.max(1, keepdims=True); e = np.exp(zz)
                res.append(e / e.sum(1, keepdims=True)); s += h
            outs.append(res)
        return [np.concatenate([o[k] for o in outs]) if outs else np.zeros((0, HEADS[k]), np.float32) for k in range(3)]


import os   # noqa: E402  (dùng trong EdgeCNN)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "export":
        export(sys.argv[1])
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
