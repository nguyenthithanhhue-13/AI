"""Bộ phân loại VAI (đích / điểm ghé / gây nhiễu / mốc) đọc CẢ CÂU (đợt 18).

Dùng chính mô hình e5-small (multilingual-e5-small, 118 triệu tham số, đang dùng cho các phần khác) với bộ chuyển LoRA
nhỏ (~0,3 triệu tham số) + một lớp phân loại: tổng tham số khi dự đoán vẫn < 200 triệu (đề mục 8.3).
Đầu vào: câu con chứa địa điểm đang xét, tên được bao bởi « », nối với toàn bộ yêu cầu.
Học từ train (+ validation cho bản final) với tăng cường: thay tên bằng tên trong kho kiến thức, bỏ dấu, gõ sai.
"""
import math, random, re
from pathlib import Path

import numpy as np

BASE = Path("F:/hf_cache/e5_small_pt")
ROLES4 = ["goal", "via", "dis", "anc"]


def marked_input(me, full_text):
    """Câu con có dấu (giữ dấu phẩy / hai chấm), tên đang xét bao bởi « », nối với cả yêu cầu."""
    toks, idx = me["sentp_acc"], me["idx"]
    a, b = idx[me["i"]], idx[me["j"] - 1]
    out = []
    for k, t in enumerate(toks):
        if k == a:
            out.append("«")
        out.append(t)
        if k == b:
            out.append("»")
    s = " ".join(out)
    s = re.sub(r" ([,:])", r"\1", s)
    return s + " || " + full_text


class LoRALinear:
    pass


def build_model(n_cls=4, r=8, alpha=16):
    import torch, torch.nn as nn
    from transformers import AutoModel

    class LoRA(nn.Module):
        def __init__(self, base, r, alpha):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = alpha / r

        def forward(self, x):
            return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    class RoleLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(str(BASE))
            for p in self.enc.parameters():
                p.requires_grad = False
            for layer in self.enc.encoder.layer:
                att = layer.attention.self
                att.query = LoRA(att.query, r, alpha)
                att.value = LoRA(att.value, r, alpha)
            h = self.enc.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * h, n_cls))

        def forward(self, ids, mask, span):
            hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            cls = hs[:, 0]
            sp = (hs * span.unsqueeze(-1)).sum(1) / span.sum(1, keepdim=True).clamp(min=1)
            return self.head(torch.cat([cls, sp], -1))

        def trainable(self):
            return [p for p in self.parameters() if p.requires_grad]

    return RoleLM()


class Featurizer:
    def __init__(self, max_len=160):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(BASE))
        self.max_len = max_len

    def __call__(self, texts):
        enc = self.tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="np")
        ids, mask, off = enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
        span = np.zeros(ids.shape, np.float32)
        for n, t in enumerate(texts):
            q = "query: " + t
            a, b = q.find("«"), q.find("»")
            if a < 0 or b < 0:
                continue
            for k, (s, e) in enumerate(off[n]):
                if e > s and s >= a and e <= b + 1:
                    span[n, k] = 1.0
        return ids, mask, span


class RoleLMPredictor:
    """Dùng khi dự đoán: nạp bộ chuyển LoRA + lớp phân loại đã học."""

    def __init__(self, path, device=None):
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = build_model()
        sd = torch.load(path, map_location="cpu")
        self.m.load_state_dict(sd, strict=False)
        self.m.to(self.dev).eval()
        self.f = Featurizer()
        self.cache = {}

    def predict(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            ids, mask, span = self.f(chunk)
            with self.torch.no_grad():
                lo = self.m(self.torch.tensor(ids).to(self.dev), self.torch.tensor(mask).to(self.dev),
                            self.torch.tensor(span).to(self.dev))
                pr = self.torch.softmax(lo, -1).cpu().numpy()
            for t, x in zip(chunk, pr):
                self.cache[t] = x
        return np.array([self.cache[t] for t in texts])
