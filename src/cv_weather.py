"""CNN đọc biểu tượng thời tiết (mưa / khô), thay MLP `weather`: ảnh cắt 40x40x3 (ảnh thu nhỏ 1/2, phủ 80px gốc).

    python src/cv_weather.py dev        # học train, đo validation   -> outputs/models_dev/weather_cnn.onnx
    python src/cv_weather.py final      # học train + validation     -> outputs/models_final/weather_cnn.onnx
"""
import os
import sys
import time

import numpy as np

from common import *


def prep(x):
    return x.reshape(-1, 40, 40, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - 0.5


def build_net():
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    return nn.Sequential(block(3, 24), nn.MaxPool2d(2), block(24, 48), nn.MaxPool2d(2), block(48, 96), nn.MaxPool2d(2),
                         block(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 2))


def main(mode, epochs=10):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / "cvdata_train.npz"); X, Y = d["we_x"], d["we_y"]
    dv = np.load(CACHE / "cvdata_validation.npz"); Xv, Yv = dv["we_x"], dv["we_y"]
    X, Y = with_strong(X, Y, "we", mode, extra=[(Xv, Yv)] if mode.startswith("final") else [])
    net = build_net()
    init_from(net, "weather_cnn.pt", mode)
    net = net.to(dev)
    print(f"CNN thời tiết ({dev}): {len(X)} mẫu, {sum(p.numel() for p in net.parameters()):,} tham số", flush=True)
    bs = 128
    steps = epochs * (len(X) // bs + 1)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(os.environ.get("MAX_LR", "3e-3")), total_steps=steps)
    Xg = torch.from_numpy(X); Yg = torch.from_numpy(Y.astype(np.int64))
    Xvg = torch.from_numpy(Xv).to(dev); Yvg = torch.from_numpy(Yv.astype(np.int64)).to(dev)   # nhỏ (1800 mẫu)

    def to_float(xb):
        return xb.view(-1, 40, 40, 3).permute(0, 3, 1, 2).float() / 255.0 - 0.5

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            if len(idx) < 32:
                continue          # lô cuối quá nhỏ: BatchNorm trên vài mẫu làm gradient nhảy vọt (xem cv_edge.py)
            xb = to_float(Xg[idx].to(dev)); yb = Yg[idx].to(dev)
            # dịch ngẫu nhiên vài pixel + đổi độ sáng nhẹ
            dx, dy = np.random.randint(-3, 4, 2)
            xb = torch.roll(xb, shifts=(int(dy), int(dx)), dims=(2, 3))
            xb = xb * (1 + 0.15 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)) + 0.06 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)
            loss = F.cross_entropy(net(xb), yb, label_smoothing=0.02)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            err = int((net(to_float(Xvg)).argmax(1) != Yvg).sum())
        net.train()
        print(f"   epoch {ep + 1}/{epochs} loss {loss.item():.4f} lỗi validation {err}/{len(Xv)} ({time.time() - t0:.0f}s)", flush=True)
    folder = OUT / f"models_{mode}"
    folder.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, folder / "weather_cnn.pt")
    export(mode)


def export(mode):
    import torch
    net = build_net()
    net.load_state_dict(torch.load(OUT / f"models_{mode}" / "weather_cnn.pt"))
    net.eval()
    out = OUT / f"models_{mode}" / "weather_cnn.onnx"
    torch.onnx.export(net, torch.zeros(2, 3, 40, 40), str(out), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    (OUT / f"models_{mode}" / "weather_cnn_params.txt").write_text(str(sum(p.numel() for p in net.parameters())))
    print("đã lưu", out)


class WeatherCNN:
    """Cùng giao diện với MLP: predict_proba(X uint8) -> [mảng (n, 2) xác suất khô / mưa]."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        p = str(path).replace("weather_cnn.onnx", "weather_cnn_params.txt")
        self.n_params = int(open(p).read()) if os.path.exists(p) else 0

    def predict_proba(self, X):
        z = self.sess.run(None, {"x": prep(np.asarray(X))})[0]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return [e / e.sum(1, keepdims=True)]


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
