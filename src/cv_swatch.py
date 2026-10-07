"""CNN đọc một dòng chú giải, thay MLP `swatch`.

Ảnh cắt (6688 byte) gồm ba phần, mỗi phần đi qua một nhánh conv riêng rồi ghép lại:
  - mẫu ký hiệu màu 36x24x3 (ảnh 1/2)      -> nhánh A
  - chữ bên phải 144x16 xám (chuẩn hóa cỡ) -> nhánh B
  - mẫu ở độ phân giải gốc 64x28 xám       -> nhánh C
Ba đầu ra: ý nghĩa dòng (19 lớp), kiểu nét của mẫu (3), chữ thời tiết (3).

    python src/cv_swatch.py dev     # học train, đo validation -> outputs/models_dev/swatch_cnn.onnx
    python src/cv_swatch.py final   # học train + validation   -> outputs/models_final/swatch_cnn.onnx
"""
import os
import sys
import time

import numpy as np

from common import *

HEADS = (19, 3, 3)
NA, NB, NC = 36 * 24 * 3, 144 * 16, 64 * 28


def split_parts(x):
    """uint8 (n, 6688) -> (mẫu màu (n,3,24,36), chữ (n,1,16,144), mẫu gốc (n,1,28,64)), đều trong [-0.5, 0.5]."""
    x = np.asarray(x)
    a = x[:, :NA].reshape(-1, 24, 36, 3).transpose(0, 3, 1, 2).astype(np.float32) / 255.0 - 0.5
    b = x[:, NA:NA + NB].reshape(-1, 1, 16, 144).astype(np.float32) / 255.0 - 0.5
    c = x[:, NA + NB:].reshape(-1, 1, 28, 64).astype(np.float32) / 255.0 - 0.5
    return a, b, c


def build_net():
    import torch
    import torch.nn as nn

    def blk(a, b, pool=True):
        layers = [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                  nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = nn.Sequential(blk(3, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())        # mẫu màu
            self.B = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 96), nn.AdaptiveAvgPool2d((1, 4)), nn.Flatten())   # chữ
            self.C = nn.Sequential(blk(1, 32), blk(32, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())        # mẫu gốc
            self.head = nn.Sequential(nn.Linear(64 + 96 * 4 + 64, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                      nn.Linear(256, sum(HEADS)))

        def forward(self, a, b, c):
            return self.head(torch.cat([self.A(a), self.B(b), self.C(c)], 1))

    return Net()


def main(mode, epochs=12):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(CACHE / "cvdata_train.npz"); X, Y = d["sw_x"], d["sw_y"]
    dv = np.load(CACHE / "cvdata_validation.npz"); Xv, Yv = dv["sw_x"], dv["sw_y"]
    X, Y = with_strong(X, Y, "sw", mode, extra=[(Xv, Yv)] if mode.startswith("final") else [])
    net = build_net()
    init_from(net, "swatch_cnn.pt", mode)
    net = net.to(dev)
    print(f"CNN chú giải ({dev}): {len(X)} mẫu, {sum(p.numel() for p in net.parameters()):,} tham số", flush=True)
    bs = 192
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(os.environ.get("MAX_LR", "3e-3")), total_steps=epochs * (len(X) // bs + 1))
    # GPU 4 GB: giữ dữ liệu ở RAM, chuyển từng lô
    Xg = torch.from_numpy(X); Yg = torch.from_numpy(Y.astype(np.int64))
    Xvg = torch.from_numpy(Xv); Yvg = torch.from_numpy(Yv.astype(np.int64))

    def parts(xb):
        a = xb[:, :NA].view(-1, 24, 36, 3).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        b = xb[:, NA:NA + NB].view(-1, 1, 16, 144).float() / 255.0 - 0.5
        c = xb[:, NA + NB:].view(-1, 1, 28, 64).float() / 255.0 - 0.5
        return a, b, c

    def heads(z):
        out, s = [], 0
        for h in HEADS:
            out.append(z[:, s:s + h]); s += h
        return out

    def evaluate():
        net.eval(); err = [0, 0, 0]
        with torch.no_grad():
            for i in range(0, len(Xvg), 1024):
                z = heads(net(*parts(Xvg[i:i + 1024].to(dev))))
                for k in range(3):
                    y = Yvg[i:i + 1024, k].to(dev); v = y >= 0
                    err[k] += (z[k].argmax(1)[v] != y[v]).sum().item()
        net.train()
        return err

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            if len(idx) < 32:
                continue          # lô cuối quá nhỏ: BatchNorm trên vài mẫu làm gradient nhảy vọt (xem cv_edge.py)
            xb = Xg[idx].to(dev); yb = Yg[idx].to(dev)
            a, b, c = parts(xb)
            g = 1 + 0.12 * (torch.rand(len(a), 1, 1, 1, device=dev) - 0.5)       # đổi độ sáng / tương phản nhẹ
            a, b, c = a * g, b * g, c * g
            z = heads(net(a, b, c))
            # đầu ra nào không có nhãn hợp lệ trong lô (vd chữ thời tiết ở lô cuối nhỏ) thì bỏ qua: tránh loss = nan
            loss = sum(F.cross_entropy(z[k], yb[:, k], label_smoothing=0.02, ignore_index=-1)
                       for k in range(3) if (yb[:, k] >= 0).any())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"   epoch {ep + 1}/{epochs} loss {loss.item():.4f} lỗi validation (ý nghĩa / kiểu nét / chữ thời tiết) {evaluate()} ({time.time() - t0:.0f}s)", flush=True)
    folder = OUT / f"models_{mode}"
    folder.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, folder / "swatch_cnn.pt")
    export(mode)


def export(mode):
    import torch
    net = build_net()
    net.load_state_dict(torch.load(OUT / f"models_{mode}" / "swatch_cnn.pt"))
    net.eval()
    out = OUT / f"models_{mode}" / "swatch_cnn.onnx"
    torch.onnx.export(net, (torch.zeros(2, 3, 24, 36), torch.zeros(2, 1, 16, 144), torch.zeros(2, 1, 28, 64)), str(out),
                      input_names=["a", "b", "c"], output_names=["logits"],
                      dynamic_axes={"a": {0: "n"}, "b": {0: "n"}, "c": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=True)      # AdaptiveAvgPool2d((1, 4)) trên bản đồ đặc trưng rộng 18: trình xuất cũ không hỗ trợ
    (OUT / f"models_{mode}" / "swatch_cnn_params.txt").write_text(str(sum(p.numel() for p in net.parameters())))
    print("đã lưu", out)


class SwatchCNN:
    """Cùng giao diện với MLP: predict_proba(X uint8) -> [ý nghĩa dòng, kiểu nét mẫu, chữ thời tiết]."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        p = str(path).replace("swatch_cnn.onnx", "swatch_cnn_params.txt")
        self.n_params = int(open(p).read()) if os.path.exists(p) else 0

    def predict_proba(self, X, batch=2048):
        outs = []
        for i in range(0, len(X), batch):
            a, b, c = split_parts(X[i:i + batch])
            z = self.sess.run(None, {"a": a, "b": b, "c": c})[0]
            res, s = [], 0
            for h in HEADS:
                zz = z[:, s:s + h]; zz = zz - zz.max(1, keepdims=True); e = np.exp(zz)
                res.append(e / e.sum(1, keepdims=True)); s += h
            outs.append(res)
        return [np.concatenate([o[k] for o in outs]) if outs else np.zeros((0, HEADS[k]), np.float32) for k in range(3)]


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
