"""Tìm cụm chỉ địa điểm trong câu DỰA VÀO NGỮ CẢNH, để nhận ra cả những tên gọi chưa từng gặp.

Ý tưởng: tên gọi địa điểm là "từ mở" (có thể mới), còn các từ khung câu quanh nó ("tới", "ghé", "ở", "trước", "rồi",
"giúp mình"...) là "từ đóng" lặp đi lặp lại. Mỗi từ được phân loại "thuộc cụm địa điểm hay không" bằng hồi quy logistic
trên: các từ khung lân cận (từ hiếm bị thay bằng UNK), từ mở đầu kiểu "phòng / khu / nơi / nhà...", và dấu câu.
Khi huấn luyện, chính các từ của tên địa điểm bị che ngẫu nhiên thành UNK để mô hình buộc phải học ngữ cảnh.
"""
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# từ thường mở đầu một tên địa điểm (kiến thức tiếng Việt chung, không lấy từ test)
HEADS = {"phong", "khu", "noi", "nha", "san", "bai", "toa", "day", "cho", "tram", "cong", "loi", "ham", "xuong",
         "quay", "kho", "ban", "van", "diem", "sanh", "hoi", "lop", "bep", "chot", "thu", "can", "ky", "giang", "lab",
         "canteen", "ktx", "trung", "tang", "vien", "quan", "hanh"}


def tag_tokens(s):
    """Tách từ, GIỮ dấu phẩy và hai chấm (là ranh giới cụm rất có ích)."""
    return re.findall(r"[a-z0-9]+|[,:]", s)


class SpanTagger:
    def __init__(self, min_freq=8):
        self.min_freq = min_freq

    def _f(self, tok, drop=False):
        if drop or tok not in self.vocab:
            return "UNK"
        return tok

    def _feats(self, toks, i, dropped):
        f = {}
        n = len(toks)
        g = lambda k: ("BOS" if i + k < 0 else "EOS") if not (0 <= i + k < n) else self._f(toks[i + k], dropped[i + k])
        for k in range(-3, 4):
            f[f"w{k}={g(k)}"] = 1
        f[f"b-2-1={g(-2)}_{g(-1)}"] = 1
        f[f"b-10={g(-1)}_{g(0)}"] = 1
        f[f"b01={g(0)}_{g(1)}"] = 1
        f[f"b12={g(1)}_{g(2)}"] = 1
        f[f"t-101={g(-1)}_{g(0)}_{g(1)}"] = 1
        for k in (-2, -1, 0, 1):
            if 0 <= i + k < n and toks[i + k] in HEADS:
                f[f"head{k}"] = 1
        f["pos_from_end"] = min(n - 1 - i, 6) / 6.0
        f["pos_from_start"] = min(i, 6) / 6.0
        return f

    def fit(self, sents, spans_list, seed=0):
        """sents: list các danh sách từ (tag_tokens); spans_list: list các [(start, end)] là cụm địa điểm."""
        rng = np.random.default_rng(seed)
        cnt = Counter(t for s in sents for t in s)
        # CHỈ các từ rất phổ biến (từ khung câu, từ mở đầu như "phòng", "khu") mới được giữ nguyên dạng; mọi từ hiếm hơn
        # luôn là UNK. Nhờ vậy một tên địa điểm có trong dữ liệu nhưng chưa vào từ điển (nhãn = "không phải địa điểm")
        # không bị mô hình học thuộc lòng là "không phải địa điểm".
        thr = max(100, int(0.02 * len(sents)))
        self.vocab = {t for t, c in cnt.items() if c >= thr}
        X, y = [], []
        for toks, spans in zip(sents, spans_list):
            lab = [0] * len(toks)
            for a, b in spans:
                for k in range(a, b):
                    lab[k] = 1
            for copy in range(2):
                dropped = [False] * len(toks)
                if copy > 0:
                    for a, b in spans:          # che cả tên địa điểm (giả lập tên lạ): giữ từ đầu với xác suất 0.5
                        for k in range(a, b):
                            dropped[k] = not (k == a and rng.random() < 0.5)
                for i in range(len(toks)):
                    X.append(self._feats(toks, i, dropped)); y.append(lab[i])
        self.vec = DictVectorizer()
        self.clf = LogisticRegression(C=3, max_iter=2000).fit(self.vec.fit_transform(X), y)
        return self

    def predict(self, toks):
        if not toks:
            return np.zeros(0)
        dropped = [False] * len(toks)
        X = self.vec.transform([self._feats(toks, i, dropped) for i in range(len(toks))])
        return self.clf.predict_proba(X)[:, 1]

    def spans(self, toks, known=(), thr=0.5):
        """Các cụm địa điểm CHƯA nằm trong `known` (các khoảng đã khớp từ điển). Dấu câu không thuộc cụm."""
        p = self.predict(toks)
        used = [False] * len(toks)
        for a, b in known:
            for k in range(a, b):
                used[k] = True
        out, i = [], 0
        while i < len(toks):
            if p[i] > thr and not used[i] and toks[i] not in (",", ":"):
                j = i
                while j < len(toks) and p[j] > thr * 0.8 and not used[j] and toks[j] not in (",", ":"):
                    j += 1
                if j - i <= 7:
                    out.append((i, j))
                i = j
            else:
                i += 1
        return out
