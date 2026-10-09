"""(vòng private test) Bộ đọc yêu cầu vòng 2 = mô tả nơi giao qua bản đồ (src/mapref.py) + MissionParser2 (src/nlp2.py)
học lại trên train mới.

Khi học: câu có nơi giao mô tả qua bản đồ được THAY cụm mô tả bằng tên chuẩn của loại đích (lấy từ nhãn), rồi học như cũ.
Khi đọc: tìm vị trí đích trên bản đồ (do CV đọc) theo mô tả, thay cụm mô tả bằng tên loại địa điểm ở vị trí đó, đọc phần
còn lại (điểm ghé, địa điểm gây nhiễu, gấp, dễ vỡ) bằng MissionParser2, rồi CHỐT đích đúng vị trí (goal_ref = ("pos", (r, c))).
Mốc của "gần X nhất" được đọc bằng chính MissionParser2 (tên lạ đoán bằng từ điển / e5) và chỉ xét các loại có MỘT bản trên
bản đồ (trên train + validation mốc luôn có đúng một bản).
"""
import pickle
import re

import mapref
from nlp2 import MissionParser2, resolve_with_map, CANONICAL_ACC, PLACE_TYPES, _unaccent


def _name(t, accented):
    v = CANONICAL_ACC[t]
    return v if accented else _unaccent(v)


def _subst(text, span, t):
    a, b = span
    return text[:a] + _name(t, not text.isascii()) + text[b:]


# (vòng private) Khung ĐỐI CHIẾU trong vế giao hàng: "đưa A tới không phải X mà là Y giúp mình", "đưa A tới Y chứ không phải X".
# Bộ đọc coi cả vế có "không phải" là phủ định -> mất đích Y (và hay đôn điểm ghé lên làm đích: lỗi ĐÍCH + GHÉ trên chính train).
# Viết lại thành câu khẳng định + câu gây nhiễu quen thuộc: "đưa A tới Y giúp mình. X không phải điểm nhận."
CONTRAST_FIX = False      # TẮT: p22 (có viết lại) LB 0,8317 < p19 0,8350 (62 dòng khác, 48 dòng do viết lại) -> hại trên test
_KP = r"(?:khong|kong|khng|hkong|kohng|khogn|khog)\s+(?:phai|hai|pahi|phi|pai|phia|phair|phaii)"
_END = r"(?=\s*(?:[.,;:!?)(]|$)|\s+(?:giup|nhe|nha|dang|nho|truoc|roi|nhung)\b)"


def contrast_rewrite(text):
    keep = mapref._unaccent_keep(text)
    if len(keep) != len(text):
        return text
    m = re.search(r"\b" + _KP + r"\s+(.{2,60}?)\s+ma\s+la\s+", keep)
    if m and not re.match(r"(o do|diem nhan|la)\b", m.group(1)):
        x = text[m.start(1):m.end(1)]
        out = text[:m.start()] + text[m.end():]
        e = re.search(r"[.!?]", out[m.start():])
        cut = m.start() + (e.end() if e else len(out) - m.start())
        return out[:cut].rstrip() + (" " if e else ". ") + x + " không phải điểm nhận. " + out[cut:].lstrip()
    m = re.search(r"\s*,?\s*\bchu\s+" + _KP + r"\s+(.{2,60}?)" + _END, keep)
    if m and not re.match(r"(o do|diem nhan)\b", m.group(1)):
        x = text[m.start(1):m.end(1)]
        out = text[:m.start()] + text[m.end():]
        e = re.search(r"[.!?]", out[m.start():])
        cut = m.start() + (e.end() if e else len(out) - m.start())
        return out[:cut].rstrip() + (" " if e else ". ") + x + " không phải điểm nhận. " + out[cut:].lstrip()
    return text


def is_mapref_mission(m):
    g = m.get("goal_ref")
    return bool(g) and (g["kind"] in mapref.KINDS_MOST or g["kind"] == "anchor_near")


class MissionParser3:
    def fit(self, missions, verbose=False):
        self.vocab = mapref.build_vocab([m["text"] for m in missions])
        mapref.VOCAB = self.vocab
        ms, skipped = [], 0
        for m in missions:
            if is_mapref_mission(m):
                f = mapref.find(m["text"], self.vocab)
                if not f or not f["span"]:
                    skipped += 1
                    continue
                ms.append(dict(m, text=_subst(m["text"], f["span"], m["goal"]), goal_ref=None))
            else:
                ms.append(m)
        if verbose:
            print("nlp3: câu mô tả qua bản đồ không tìm được cụm (bỏ khi học):", skipped)
        # Từ điển tên gọi = tên khai thác từ dữ liệu + TÊN CHUẨN của 10 loại + kho tên gọi viết tay (nlp_knowledge, soạn từ kiến
        # thức chung ở vòng 1). Dữ liệu mới không dùng tên chuẩn "nhà thể thao" nên trước đây cụm này không được nhận là địa điểm
        # (phép thử tự soạn scratch/p2_n5_probe.py: mọi ca sai đều dính tên này). Tên thuộc nhiều loại bị bỏ.
        import nlp2 as _n2
        from nlp import norm, tokens
        from collections import defaultdict
        from nlp_knowledge import PLACES
        # LEX_KB: "canon" = chỉ 10 tên chuẩn (mặc định); "full" = cả kho viết tay. Bản dùng cả kho (p6) tốt hơn trên validation
        # nhưng thấp hơn 1,6 điểm trên bảng xếp hạng (kho có tên chỉ NGƯỜI như "thủ thư", "đội bóng" dễ bị hiểu thành địa điểm).
        self.lex_kb = getattr(self, "lex_kb", __import__("os").environ.get("NLP_LEX", "canon"))
        PERSONS = {"thủ thư", "đội bóng", "bếp trưởng", "lớp trưởng", "văn thư"}     # tên chỉ NGƯỜI, không phải địa điểm
        kb = defaultdict(set)
        for t, lst in PLACES.items():
            extra = lst if self.lex_kb == "full" else [a for a in lst if a not in PERSONS] if self.lex_kb == "places" else []
            for a in extra + [CANONICAL_ACC[t]]:
                kb[" ".join(tokens(norm(a)))].add(t)
        kb = {a: next(iter(ts)) for a, ts in kb.items() if a and len(ts) == 1}
        orig = _n2.mine_lexicon
        _n2.mine_lexicon = lambda missions: {**kb, **orig(missions)}
        try:
            self.p2 = MissionParser2().fit(ms, verbose=verbose)
        finally:
            _n2.mine_lexicon = orig
        return self

    def type_anchor(self, text, f, one):
        """Loại của mốc trong "gần X nhất": đọc cụm X như một nơi giao, chỉ xét các loại có một bản trên bản đồ."""
        a, b = f["anchor_span"]
        mini = ("Giao hàng tới " if not text.isascii() else "Giao hang toi ") + text[a:b] + "."
        r = self.p2.parse(mini, set(one))
        if r.get("goal_known") and r["goal"] in one:
            return r["goal"]
        d = r["goal_dist"]
        return max(one, key=lambda t: d[PLACE_TYPES.index(t)])

    def parse(self, text, present=None, landmarks=None):
        if CONTRAST_FIX:
            text = contrast_rewrite(text)
        mapref.VOCAB = self.vocab
        f = mapref.find(text, self.vocab)
        if f and f["span"] and landmarks:
            at = None
            if f["kind"] == "anchor_near":
                one = [t for t, v in landmarks.items() if len(v) == 1]
                at = self.type_anchor(text, f, one) if one else None
            pos = mapref.resolve(f["kind"], landmarks, at) if (f["kind"] != "anchor_near" or at) else None
            if pos:
                gt = next(t for t, v in landmarks.items() if pos in v)
                r = dict(self.p2.parse(_subst(text, f["span"], gt), present))
                r.update(goal=gt, goal_ref=("pos", pos), goal_known=True, mapref=(f["kind"], at, pos))
                return r
        r = dict(self.p2.parse(text, present))
        if f:
            r["mapref"] = (f["kind"], None, None)
        return r

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


def resolve3(m, landmarks, world=None):
    """Như resolve_with_map; đích đã chốt theo vị trí ("pos") được giữ nguyên."""
    if m.get("goal_ref") and m["goal_ref"][0] == "pos":
        pos = m["goal_ref"][1]
        out = resolve_with_map(dict(m, goal_ref=None, goal_known=True), landmarks, world)
        out["goal"], out["goal_ref"] = m["goal"], ("pos", pos)
        return out
    return resolve_with_map(m, landmarks, world)
