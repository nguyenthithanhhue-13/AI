"""Đọc yêu cầu tiếng Việt -> {goal, goal_ref, via, via_ref, urgent, fragile}.

Tất cả đều HỌC từ dữ liệu có nhãn (train, và validation khi huấn luyện bản cuối), không dùng test:
  1. Từ điển tên gọi địa điểm (alias -> loại) khai thác tự động từ các khung câu gây nhiễu + nhãn goal/via.
  2. Tìm các lần nhắc địa điểm trong câu (so khớp mờ để chịu lỗi gõ, câu không dấu).
  3. Bộ phân loại vai trò của từng lần nhắc: goal / via / gây nhiễu / mốc (anchor)  (TF-IDF + hồi quy logistic).
  4. Bộ phân loại tham chiếu không gian dựa trên vài từ đứng sau lần nhắc.
  5. Bảng cụm từ gấp / dễ vỡ học từ các câu không chứa địa điểm + bộ phân loại ký tự dự phòng.
  6. Dự phòng khi không tìm thấy tên địa điểm nào quen: phân loại cả câu (TF-IDF ký tự) -> loại đích.
"""
import pickle
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PREFIX = r"^(robot oi|yeu cau moi|nho ban nhe|xin chao|chao robot|nhan robot)\b[,: ]*"
DIS_FRAMES = [r"^khong can ghe (.+)$", r"^dung nham voi (.+) nhe$", r"^hom qua da giao o (.+) roi$",
              r"^nguoi nhan da roi (.+) roi$", r"^(.+) khong phai diem nhan$", r"^bo qua (.+), khong phai o do$"]
ROLES = ["goal", "via", "dis", "anc"]
KINDS = ["none", "north", "south", "west", "east", "near", "far"]
PLACE_TYPES = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]


def norm(t):
    t = t.lower().replace("đ", "d")
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def sentences(text):
    t = re.sub(r"\[don #\d+\]", " ", norm(text))
    out = []
    for p in re.split(r"[.!?]+", t):
        p = re.sub(r"\s+", " ", p).strip(" ,")
        p = re.sub(PREFIX, "", p).strip(" ,")
        if p:
            out.append(p)
    return out


def tokens(s):
    return re.findall(r"[a-z0-9]+", s)


def dl_distance(a, b, maxd=2):
    """Khoảng cách Damerau-Levenshtein (có tính hoán vị hai ký tự kề nhau)."""
    if abs(len(a) - len(b)) > maxd:
        return maxd + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


FUZZY_COMMON = 100   # từ gặp >= 100 lần trong dữ liệu học không được coi là bản gõ sai của tên gọi; None = tắt


class Lexicon:
    """alias -> loại, có so khớp mờ theo từng từ.

    Một cụm lệch chính tả chỉ được nhận nếu bản thân cụm đó KHÔNG phải cụm quen thuộc trong tập huấn luyện
    (ví dụ 'cong van' xuất hiện nhiều lần như một cụm hợp lệ nên không bị coi là 'cong vao' gõ sai).
    """

    def __init__(self, alias2type, ngram_freq):
        self.alias2type = dict(alias2type)
        self.ngram_freq = ngram_freq
        self.by_len = {}
        for a in self.alias2type:
            self.by_len.setdefault(len(a.split()), []).append((a, a.split()))
        self._tok_cache = {}
        self._find_cache = {}

    def _tok_dist(self, w, a):
        """Khoảng cách giữa hai từ (0, 1 hoặc 9 nếu khác quá xa); có lọc nhanh theo ký tự đầu."""
        key = (w, a)
        d = self._tok_cache.get(key)
        if d is None:
            if len(a) < 2 or abs(len(w) - len(a)) > 1 or not (w[0] == a[0] or w[:1] == a[1:2] or w[1:2] == a[:1]):
                d = 9
            else:
                d = dl_distance(w, a, 1)
                d = d if d <= 1 else 9
            self._tok_cache[key] = d
        return d

    def _fuzzy(self, wt, at):
        total = 0
        for w, a in zip(wt, at):
            if w == a:
                continue
            d = self._tok_dist(w, a)
            if d > 1:
                return None
            # lỗi gõ thật tạo ra từ hiếm; từ lệch mà là từ thông dụng ("ben trong" ~ "bep truong") thì không phải gõ sai
            if FUZZY_COMMON is not None and self.ngram_freq.get(w, 0) >= FUZZY_COMMON:
                return None
            total += d
        n = sum(len(a) for a in at)
        if total > (1 if n < 9 else 2) or n < 5:
            return None
        if self.ngram_freq.get(" ".join(wt), 0) >= 3:
            return None
        return total

    def find(self, toks):
        """Trả về danh sách (start, end, type, alias) không chồng nhau, ưu tiên khớp đúng và alias dài."""
        key = tuple(toks)
        if key in self._find_cache:
            return self._find_cache[key]
        cands = []
        n = len(toks)
        for k, aliases in self.by_len.items():
            for i in range(n - k + 1):
                wt = toks[i:i + k]
                for a, at in aliases:
                    if wt == at:
                        cands.append((0, -len(a), i, i + k, a))
                    else:
                        d = self._fuzzy(wt, at)
                        if d is not None:
                            cands.append((d, -len(a), i, i + k, a))
        cands.sort()
        used = [False] * n
        out = []
        for d, _, i, j, a in cands:
            if any(used[i:j]):
                continue
            for t in range(i, j):
                used[t] = True
            out.append((i, j, self.alias2type[a], a))
        out.sort()
        self._find_cache[key] = out
        return out


def ngram_counts(missions, maxn=6):
    c = Counter()
    for m in missions:
        for s in sentences(m["text"]):
            t = tokens(s)
            for k in range(1, maxn + 1):
                for i in range(len(t) - k + 1):
                    c[" ".join(t[i:i + k])] += 1
    return {k: v for k, v in c.items() if v >= 3}


def mine_lexicon(missions):
    """missions: list dict có text, goal, via, goal_ref, via_ref (dạng scenes.json)."""
    alias_cnt = Counter()
    rest = []
    for m in missions:
        types = {m["goal"]}
        if m["via"]:
            types.add(m["via"])
        for k in ("goal_ref", "via_ref"):
            if m[k] and m[k]["anchor"]:
                types.add(m[k]["anchor"])
        keep = []
        for x in sentences(m["text"]):
            for pat in DIS_FRAMES:
                mm = re.match(pat, x)
                if mm:
                    alias_cnt[mm.group(1)] += 1
                    break
            else:
                # câu gây nhiễu bị gõ sai sẽ không khớp khung -> lọc lỏng bằng từ khóa để không làm bẩn thống kê
                if not re.search(r"khong can ghe|dung nham|hom qua da|nguoi nhan da|diem nhan|bo qua|o do$", x):
                    keep.append(x)
        rest.append((" . ".join(keep), types, m["goal"]))
    lex = {}
    for a, n in alias_cnt.items():
        pat = re.compile(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])")
        tc = Counter()
        tot = 0
        for text, types, _ in rest:
            if pat.search(text):
                tot += 1
                for t in types:
                    tc[t] += 1
        if tot >= 2:
            top = tc.most_common(2)
            second = top[1][1] if len(top) > 1 else 0
            if top[0][1] >= 0.75 * tot and top[0][1] >= second + 2:
                lex[a] = top[0][0]
    # người nhận: "<người nhận> (o <địa điểm>)? (dang cho|can nhan) <hàng>"
    rc = defaultdict(Counter)
    place_pat = "|".join(sorted(map(re.escape, lex), key=len, reverse=True))
    for text, types, goal in rest:
        for x in text.split(" . "):
            mm = re.match(r"^(.+?) (dang cho|can nhan) ", x)
            if not mm:
                continue
            who = re.sub(r" o (" + place_pat + r").*$", "", mm.group(1)).strip()
            if who and len(who.split()) <= 6 and not re.search(place_pat, who):
                rc[who][goal] += 1
    for who, c in rc.items():
        t, n = c.most_common(1)[0]
        if sum(c.values()) >= 2 and n >= 0.9 * sum(c.values()) and who not in lex:
            lex[who] = t
    return lex


_DIS_TOK = [re.sub(r"[,:]", "", p).replace("(.+)", ".+") for p in DIS_FRAMES]


class MissionParser:
    def __init__(self):
        self.lex = None

    # ---------- đặc trưng ----------
    def _mentions(self, text):
        """-> list (sent_idx, toks, i, j, type, mentions của câu); và danh sách câu không có địa điểm."""
        out, plain = [], []
        for si, s in enumerate(sentences(text)):
            toks = tokens(s)
            ms = self.lex.find(toks)
            if not ms:
                plain.append(s)
            for (i, j, t, a) in ms:
                out.append((si, toks, i, j, t, ms))
        return out, plain

    @staticmethod
    def _ctx(toks, i, j, ms):
        masked = []
        k = 0
        while k < len(toks):
            hit = next((m for m in ms if m[0] == k), None)
            if hit:
                masked.append("TGT" if hit[0] == i else "PLC")
                k = hit[1]
            else:
                masked.append(toks[k])
                k += 1
        pos = masked.index("TGT")
        left = " ".join(masked[max(0, pos - 5):pos])
        after = masked[pos + 1:pos + 6]
        right = " ".join(after)
        # ngữ cảnh cho tham chiếu không gian: cắt ngay sau địa điểm kế tiếp (mốc), giữ thêm chữ 'hon'
        ref = []
        for t_i, t in enumerate(after):
            ref.append(t)
            if t == "PLC":
                if t_i + 1 < len(after) and after[t_i + 1] == "hon":
                    ref.append("hon")
                break
        ref = ref[:5]
        order = f"n{min(len(ms), 3)}i{min([m[0] for m in ms].index(i), 2)}"
        return left, right, " ".join(masked) + " " + order, " ".join(ref)

    @staticmethod
    def _anchor_of(i, j, ms):
        """Địa điểm đứng ngay sau (cách <= 3 từ) lần nhắc (i, j) trong cùng câu."""
        nxt = [x for x in ms if x[0] >= j]
        if nxt and nxt[0][0] - j <= 3:
            return nxt[0]
        return None

    # ---------- huấn luyện ----------
    def fit(self, missions, lexicon_missions=None, verbose=False):
        lm = lexicon_missions or missions
        self.lex = Lexicon(mine_lexicon(lm), ngram_counts(lm))
        Xl, Xr, Xs, yrole = [], [], [], []
        Rr, ykind = [], []
        table = defaultdict(lambda: [0, 0, 0])
        G, yG = [], []
        for m in missions:
            ments, plain = self._mentions(m["text"])
            for s in plain:
                t = table[s]
                t[0] += 1
                t[1] += bool(m["urgent"])
                t[2] += bool(m["fragile"])
            G.append(self._fallback_text(m["text"]))
            yG.append(PLACE_TYPES.index(m["goal"]))
            roles = []
            for (si, toks, i, j, t, ms) in ments:
                role = None
                idx = [x[0] for x in ms].index(i)
                sent = " ".join(toks)
                if any(re.match(p, sent) for p in _DIS_TOK):
                    role = "dis"
                elif idx > 0:
                    p = ms[idx - 1]
                    for key in ("goal", "via"):
                        ref = m[key + "_ref"]
                        if ref and ref["anchor"] == t and p[2] == m[key] and i - p[1] <= 3 \
                                and not re.search(r"\b(ghe|qua|lay|toi|den|truoc|roi)\b", " ".join(toks[p[1]:i])):
                            role = "anc"
                if role is None:
                    role = "goal" if t == m["goal"] else ("via" if t == m["via"] else "dis")
                roles.append(role)
            for (si, toks, i, j, t, ms), role in zip(ments, roles):
                left, right, sent, refc = self._ctx(toks, i, j, ms)
                Xl.append(left); Xr.append(right); Xs.append(sent); yrole.append(ROLES.index(role))
                if role in ("goal", "via"):
                    ref = m[role + "_ref"]
                    kind = 0
                    if ref:
                        if ref["anchor"]:
                            a = self._anchor_of(i, j, ms)
                            if a is not None and a[2] == ref["anchor"]:
                                kind = KINDS.index(ref["kind"])
                        else:
                            same = [k for k, x in enumerate(ments) if x[4] == t and roles[k] == role]
                            if ments[same[-1]][2] == i and ments[same[-1]][0] == si:
                                kind = KINDS.index(ref["kind"])
                    Rr.append(refc)
                    ykind.append(kind)
        self.v_l = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.v_r = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.v_s = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"\S+", lowercase=False)
        X = hstack([self.v_l.fit_transform(["L " + x for x in Xl]), self.v_r.fit_transform(["R " + x for x in Xr]),
                    self.v_s.fit_transform(Xs)]).tocsr()
        self.role_clf = LogisticRegression(C=20, max_iter=3000).fit(X, yrole)
        self.v_k = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, lowercase=False)
        self.kind_clf = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(self.v_k.fit_transform(Rr), ykind)
        # bảng cụm gấp / dễ vỡ
        self.phrase = {}
        S, yS = [], []
        for s, (n, u, f) in table.items():
            if n >= 2:
                lab = 1 if u >= 0.9 * n and n >= 3 else (2 if f >= 0.9 * n and n >= 3 else 0)
                self.phrase[s] = lab
                S.append(s); yS.append(lab)
        self.v_p = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self.phrase_clf = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(self.v_p.fit_transform(S), yS)
        # dự phòng: cả câu -> loại đích
        self.v_g = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.goal_clf = LogisticRegression(C=10, max_iter=2000).fit(self.v_g.fit_transform(G), yG)
        if verbose:
            print("lexicon:", len(self.lex.alias2type), "aliases; role samples:", Counter(ROLES[y] for y in yrole),
                  "; kind samples:", Counter(KINDS[y] for y in ykind))
            print("phrases U:", [s for s, l in self.phrase.items() if l == 1])
            print("phrases F:", [s for s, l in self.phrase.items() if l == 2])
        return self

    @staticmethod
    def _fallback_text(text):
        keep = [s for s in sentences(text) if not any(re.match(p, re.sub(r"[,:]", "", s)) for p in _DIS_TOK)]
        return " . ".join(keep)

    # ---------- dự đoán ----------
    def _phrase_label(self, s):
        if s in self.phrase:
            return self.phrase[s]
        best, bl = 3, None
        for p, l in self.phrase.items():
            if abs(len(p) - len(s)) <= 2:
                d = dl_distance(s, p, 2)
                if d < best:
                    best, bl = d, l
        if bl is not None and best <= 2:
            return bl
        # câu ghép tiền tố lạ -> thử phần sau dấu ':' / ','
        for part in re.split(r"[:,] ", s):
            if self.phrase.get(part):
                return self.phrase[part]
        pr = self.phrase_clf.predict_proba(self.v_p.transform([s]))[0]
        k = int(np.argmax(pr))
        # chỉ tin bộ phân loại với câu ngắn và không phải câu giao hàng (câu có tên địa điểm lạ dễ bị đoán bừa)
        toks = tokens(s)
        if len(toks) > 7 or re.search(r"\b(can nhan|dang cho|giao|mang|gui|dua|chuyen|ghe|toi|den|hang:)", s):
            return 0
        return k if pr[k] > 0.6 else 0

    def parse(self, text):
        ments, plain = self._mentions(text)
        labs = [self._phrase_label(s) for s in plain]
        res = {"goal": None, "goal_ref": None, "via": None, "via_ref": None,
               "urgent": 1 in labs, "fragile": 2 in labs, "conf": 0.0, "goal_probs": None}
        gp = self.goal_clf.predict_proba(self.v_g.transform([self._fallback_text(text)]))[0]
        res["goal_probs"] = {PLACE_TYPES[c]: float(p) for c, p in zip(self.goal_clf.classes_, gp)}
        if not ments:
            res["goal"] = max(res["goal_probs"], key=res["goal_probs"].get)
            return res
        ctx = [self._ctx(toks, i, j, ms) for (si, toks, i, j, t, ms) in ments]
        X = hstack([self.v_l.transform(["L " + c[0] for c in ctx]), self.v_r.transform(["R " + c[1] for c in ctx]),
                    self.v_s.transform([c[2] for c in ctx])]).tocsr()
        P = self.role_clf.predict_proba(X)
        kindP = self.kind_clf.predict_proba(self.v_k.transform([c[3] for c in ctx]))
        g = int(np.argmax(P[:, 0]))
        res["conf"] = float(P[g, 0])
        if P[g].argmax() == 0:
            res["goal"] = ments[g][4]
        else:
            # không lần nhắc nào được coi là đích (tên đích lạ?) -> dùng bộ phân loại cả câu,
            # loại trừ các loại đã bị xếp là via / gây nhiễu
            taken = {ments[k][4] for k in range(len(ments)) if P[k].argmax() != 0}
            cand = {t: p for t, p in res["goal_probs"].items() if t not in taken}
            res["goal"] = max(cand, key=cand.get) if cand else ments[g][4]
        via_c = [k for k in range(len(ments)) if ments[k][4] != res["goal"] and P[k].argmax() == 1]
        if via_c:
            v = max(via_c, key=lambda k: P[k, 1])
            res["via"] = ments[v][4]
        for key in ("goal", "via"):
            if res[key] is None:
                continue
            role_i = ROLES.index(key)
            best = None
            for k in range(len(ments)):
                if ments[k][4] != res[key] or P[k].argmax() != role_i:
                    continue
                kk = int(np.argmax(kindP[k]))
                if kk == 0:
                    continue
                anchor = None
                if KINDS[kk] in ("near", "far"):
                    si, toks, i, j, t, ms = ments[k]
                    a = self._anchor_of(i, j, ms)
                    if a is None:
                        continue
                    anchor = a[2]
                if best is None or kindP[k, kk] > best[0]:
                    best = (kindP[k, kk], KINDS[kk], anchor)
            if best:
                res[key + "_ref"] = (best[1], best[2])
        return res

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
