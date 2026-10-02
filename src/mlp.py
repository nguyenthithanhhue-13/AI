"""Mạng nơ-ron nhiều lớp (MLP) viết bằng numpy thuần: đủ nhỏ để huấn luyện trên CPU, không cần PyTorch.

- Đầu vào: mảng uint8 (ảnh cắt đã duỗi phẳng), tự chuẩn hóa về [-0.5, 0.5].
- Nhiều "đầu ra" softmax dùng chung thân mạng (ví dụ: kiểu nét / bậc thang / một chiều của một đoạn đường).
- Tối ưu Adam theo mini-batch.
"""
import numpy as np


class MLP:
    def __init__(self, n_in, hidden, heads, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [sum(heads)]
        self.heads = list(heads)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    @property
    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    @staticmethod
    def _prep(X):
        return X.astype(np.float32) * (1.0 / 255.0) - 0.5

    def _forward(self, x):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                np.maximum(x, 0, out=x)
            acts.append(x)
        return acts

    def _softmax_heads(self, z):
        out = np.empty_like(z)
        s = 0
        for h in self.heads:
            zz = z[:, s:s + h]
            zz = zz - zz.max(1, keepdims=True)
            e = np.exp(zz)
            out[:, s:s + h] = e / e.sum(1, keepdims=True)
            s += h
        return out

    def predict_proba(self, X, batch=8192):
        """Trả về list mảng xác suất, mỗi phần tử ứng với một đầu ra."""
        outs = []
        for i in range(0, len(X), batch):
            outs.append(self._softmax_heads(self._forward(self._prep(X[i:i + batch]))[-1]))
        p = np.concatenate(outs) if outs else np.zeros((0, sum(self.heads)), np.float32)
        res, s = [], 0
        for h in self.heads:
            res.append(p[:, s:s + h])
            s += h
        return res

    def fit(self, X, Y, epochs=10, batch=256, lr=1e-3, wd=1e-5, seed=0, Xval=None, Yval=None, verbose=True, name="", weights=None):
        """Y: mảng int (n, số đầu ra); nhãn -1 = bỏ qua đầu ra đó cho mẫu này."""
        rng = np.random.default_rng(seed)
        Y = Y.reshape(len(Y), -1)
        n = len(X)
        m = [np.zeros_like(w) for w in self.W + self.b]
        v = [np.zeros_like(w) for w in self.W + self.b]
        t = 0
        steps_total = epochs * ((n + batch - 1) // batch)
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot_loss = 0.0
            for i in range(0, n, batch):
                idx = np.sort(perm[i:i + batch])
                x = self._prep(X[idx])
                y = Y[idx]
                acts = self._forward(x)
                p = self._softmax_heads(acts[-1])
                g = p.copy()
                s = 0
                for k, h in enumerate(self.heads):
                    yk = y[:, k]
                    valid = yk >= 0
                    rows = np.nonzero(valid)[0]
                    tot_loss += -np.log(p[rows, s + yk[rows]] + 1e-9).sum()
                    g[rows, s + yk[rows]] -= 1.0
                    g[~valid, s:s + h] = 0.0
                    if weights is not None and weights[k] is not None:
                        g[rows, s:s + h] *= weights[k][yk[rows]][:, None]
                    s += h
                g /= len(idx)
                grads_W, grads_b = [], []
                for li in range(len(self.W) - 1, -1, -1):
                    grads_W.append(acts[li].T @ g + wd * self.W[li])
                    grads_b.append(g.sum(0))
                    if li > 0:
                        g = g @ self.W[li].T
                        g[acts[li] <= 0] = 0
                grads = grads_W[::-1] + grads_b[::-1]
                t += 1
                cur_lr = lr * 0.5 * (1 + np.cos(np.pi * t / steps_total))   # giảm dần theo cosine
                params = self.W + self.b
                for j, (pp, gg) in enumerate(zip(params, grads)):
                    m[j] = 0.9 * m[j] + 0.1 * gg
                    v[j] = 0.999 * v[j] + 0.001 * gg * gg
                    pp -= cur_lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
            if verbose:
                msg = f"   [{name}] epoch {ep + 1}/{epochs} loss {tot_loss / n:.4f}"
                if Xval is not None:
                    msg += " val acc " + " ".join(f"{a:.4f}" for a in self.accuracy(Xval, Yval))
                print(msg, flush=True)
        return self

    def accuracy(self, X, Y):
        Y = Y.reshape(len(Y), -1)
        P = self.predict_proba(X)
        out = []
        for k, p in enumerate(P):
            valid = Y[:, k] >= 0
            out.append(float((p[valid].argmax(1) == Y[valid, k]).mean()) if valid.any() else float("nan"))
        return out

    def save(self, path):
        np.savez_compressed(path, heads=np.array(self.heads), n=len(self.W),
                            **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})

    @staticmethod
    def load(path):
        d = np.load(path)
        obj = MLP.__new__(MLP)
        obj.heads = [int(h) for h in d["heads"]]
        n = int(d["n"])
        obj.W = [d[f"W{i}"] for i in range(n)]
        obj.b = [d[f"b{i}"] for i in range(n)]
        return obj
