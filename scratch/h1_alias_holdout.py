"""Phép đo "giấu tên gọi" (gần với test hơn validation):
gộp train + validation, chia các tên gọi địa điểm (trừ tên chuẩn như "thư viện") thành 5 nhóm.
Lượt k: bỏ khỏi dữ liệu học MỌI câu có tên gọi thuộc nhóm k, rồi chấm trên chính các câu đó
-> mô hình phải đọc tên gọi chưa từng gặp, giống ~35% cảnh test. Chỉ dùng train/validation.

    python scratch/h1_alias_holdout.py [folds=0,1,2,3,4] [--no-knowledge] [--no-e5]
"""
import sys, re, time
sys.path.insert(0, "src")
import numpy as np
from common import *
from nlp import mine_lexicon, norm
import nlp2
from nlp2 import MissionParser2, CANONICAL
from eval_nlp2 import evaluate

args = [a for a in sys.argv[1:] if not a.startswith("--")]
folds = [int(x) for x in args[0].split(",")] if args else range(5)

_, trl, trs = load_split("train"); _, val, vas = load_split("validation")
scenes = trs + vas
labels = [trl[i * 10:(i + 1) * 10] for i in range(len(trs))] + [val[i * 10:(i + 1) * 10] for i in range(len(vas))]
missions = [s["mission"] for s in scenes]

lex = mine_lexicon(missions)
canon = {a for lst in CANONICAL.values() for a in lst}
aliases = sorted(a for a in lex if a not in canon)
rng = np.random.default_rng(42)
pats = {a: re.compile(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])") for a in aliases}
if "--phrases" in sys.argv:
    # giấu CÂU CON gấp / dễ vỡ (câu không nhắc địa điểm, gặp >= 3 lần, >= 90% cùng nhãn)
    from nlp import sentences
    allp = re.compile("|".join(sorted(map(re.escape, lex), key=len, reverse=True)))
    sents = [[x for x in sentences(m["text"]) if not allp.search(x)] for m in missions]
    tab = {}
    for m, ss in zip(missions, sents):
        for x in ss:
            t = tab.setdefault(x, [0, 0, 0]); t[0] += 1; t[1] += bool(m["urgent"]); t[2] += bool(m["fragile"])
    keys = sorted(x for x, t in tab.items() if t[0] >= 3 and (t[1] >= 0.9 * t[0] or t[2] >= 0.9 * t[0]))
    group = {a: int(g) for a, g in zip(keys, rng.permutation(len(keys)) % 5)}
    has = [set(ss) & set(keys) for ss in sents]
    print(f"{len(scenes)} cảnh, {len(keys)} câu con gấp / dễ vỡ chia 5 nhóm")
    aliases = keys
elif "--templates" in sys.argv:
    # giấu KHUNG CÂU: câu con có nhắc địa điểm, thay mọi tên gọi bằng "X" -> khung; khung gặp >= 3 lần chia 5 nhóm.
    # Lượt k bỏ khỏi dữ liệu học mọi câu có khung thuộc nhóm k -> đo cách đọc vai trò với cách diễn đạt chưa gặp.
    from nlp import sentences
    allp = re.compile(r"(?<![a-z])(" + "|".join(sorted(map(re.escape, lex), key=len, reverse=True)) + r")(?![a-z])")
    skel = lambda x: re.sub(r"\d+", "9", allp.sub("X", x))
    sents = [{skel(x) for x in sentences(m["text"]) if allp.search(x)} for m in missions]
    from collections import Counter
    tab = Counter(s for ss in sents for s in ss)
    maxc = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--maxc=")), 10 ** 9))
    keys = sorted(x for x, c in tab.items() if 3 <= c <= maxc)
    group = {a: int(g) for a, g in zip(keys, rng.permutation(len(keys)) % 5)}
    has = [ss & set(keys) for ss in sents]
    print(f"{len(scenes)} cảnh, {len(tab)} khung câu, {len(keys)} khung gặp 3..{maxc} lần chia 5 nhóm; "
          f"{sum(map(bool, has))} cảnh có ít nhất một khung được chia")
    for x, c in tab.most_common(12): print(f"   {c:5d}  {x}")
    aliases = keys
else:
    group = {a: int(g) for a, g in zip(aliases, rng.permutation(len(aliases)) % 5)}
    has = [{a for a in aliases if pats[a].search(norm(m["text"]))} for m in missions]
    print(f"{len(scenes)} cảnh, {len(aliases)} tên gọi (không tính tên chuẩn) chia 5 nhóm")

tot_n = tot_s = 0
for k in folds:
    t0 = time.time()
    held = {a for a in aliases if group[a] == k}
    ev = [i for i in range(len(scenes)) if has[i] & held]
    tr = [i for i in range(len(scenes)) if not has[i] & held]
    if "--fair" in sys.argv:
        # tên gọi bị giấu cũng phải vắng mặt trong từ khóa / ví dụ viết tay (nếu không thì đo lạc quan):
        # bỏ mọi mẫu từ khóa khớp một tên bị giấu, mọi ví dụ PLACES chứa / nằm trong một tên bị giấu
        import nlp_knowledge
        _kw0 = getattr(nlp2, "_KW0", None) or {t: list(v) for t, v in nlp2.KEYWORDS.items()}
        _pl0 = getattr(nlp2, "_PL0", None) or {t: list(v) for t, v in nlp_knowledge.PLACES.items()}
        _pm0 = getattr(nlp2, "_PM0", None) or {t: list(v) for t, v in nlp_knowledge.PLACES_MORE.items()}
        nlp2._KW0, nlp2._PL0, nlp2._PM0 = _kw0, _pl0, _pm0
        nlp2.KEYWORDS = {t: [(pt, w) for pt, w in v if not any(re.search(pt, a) for a in held)] for t, v in _kw0.items()}
        un = lambda s: norm(s)
        _pd0 = getattr(nlp2, "_PD0", None) or {t: list(v) for t, v in nlp_knowledge.PLACES_DESC.items()}
        nlp2._PD0 = _pd0
        for d, d0 in ((nlp_knowledge.PLACES, _pl0), (nlp_knowledge.PLACES_MORE, _pm0), (nlp_knowledge.PLACES_DESC, _pd0)):
            d.clear()
            d.update({t: [x for x in v if not any(a in un(x) or un(x) in a for a in held)] for t, v in d0.items()})
    p = MissionParser2()
    if "--no-knowledge" in sys.argv: p.use_knowledge = False
    if "--no-e5" in sys.argv: p.use_e5 = False
    if "--harsh" in sys.argv: p.use_knowledge = False; p.ablate_rules = True      # tên gọi thật sự xa lạ: không từ khóa, không ví dụ viết tay
    for a in sys.argv:
        if a.startswith("--ctx="): p.ctx_weight = float(a[6:])
        if a.startswith("--gw="): nlp2.GOAL_TEXT_W = float(a[5:])
        if a.startswith("--ng="): nlp2.CTX_NG_W = float(a[5:])
        if a.startswith("--nglow="): nlp2.CTX_NG_LOWCONF = float(a[8:])
        if a.startswith("--set="):                     # --set=TÊN=giá_trị: đổi một hằng số của nlp2
            name, val = a[6:].split("=")
            setattr(nlp2, name, eval(val))
    p.fit([missions[i] for i in tr])
    show = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--show=")), 0))
    sc, _, _ = evaluate(p, [scenes[i] for i in ev], [labels[i] for i in ev], show, f"nhóm {k}: học {len(tr)} chấm {len(ev)}")
    tot_n += len(ev); tot_s += sc * len(ev)
    print(f"      ({time.time() - t0:.0f} giây)", flush=True)
print(f"GIẤU TÊN GỌI: ĐIỂM (bản đồ đúng) = {tot_s / tot_n:.4f} trên {tot_n} lượt chấm")
