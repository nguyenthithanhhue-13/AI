"""Đo bộ đọc vòng 2 (nlp3) trên validation: học train -> đọc câu validation với bản đồ ĐÚNG -> so vị trí đích / điểm ghé
đã chốt với vị trí đúng, và cờ gấp / dễ vỡ.  python scratch/p2_n3_eval.py [fit]"""
import sys, time, pickle, collections
sys.path.insert(0, "src"); sys.path.insert(0, "scratch")
from common import *
from p2_lib import legs_true
import nlp3
from strategy import resolve_targets
import os
path = OUT / os.environ.get("NLPF", "nlp3_trainonly.pkl")
t = time.time()
if "fit" in sys.argv or not path.exists():
    p = nlp3.MissionParser3().fit([s["mission"] for s in load_split("train")[2]], verbose=True)
    p.save(path)
    from nlp2 import save_embed_cache; save_embed_cache()
    print(f"học xong {time.time()-t:.0f}s", flush=True)
else:
    p = nlp3.MissionParser3.load(path)
sc = load_split("validation")[2]
c = collections.Counter(); errs = collections.defaultdict(list)
for i, s in enumerate(sc):
    w = world_from_scene(s); L = w["landmarks"]
    r = p.parse(s["mission"]["text"], {t for t, v in L.items() if v}, L)
    out = nlp3.resolve3(r, L, w)
    tl = legs_true(s, w)
    pg = sorted(resolve_targets(w, out["goal"], out["goal_ref"]))
    pv = sorted(resolve_targets(w, out["via"], out["via_ref"])) if out["via"] else []
    tg = sorted(tl[-1]); tv = sorted(tl[0]) if len(tl) == 2 else []
    m = s["mission"]
    ok = dict(goal=pg == tg, via=pv == tv, urgent=out["urgent"] == m["urgent"], fragile=out["fragile"] == m["fragile"])
    ok["all"] = all(ok.values())
    kind = m["goal_ref"]["kind"] if m["goal_ref"] else "none"
    for k, v in ok.items():
        c[k] += v
        if not v and k != "all": errs[k].append((i, kind, m["text"], (m["goal"], m["goal_ref"], m["via"], m["via_ref"]), (out["goal"], out["goal_ref"], out["via"], out["via_ref"])))
    c["kind:" + kind] += 1; c["kindok:" + kind] += ok["goal"]
print({k: v for k, v in c.items() if not k.startswith("kind")}, "/", len(sc))
print("đích đúng theo kiểu:", {k[5:]: f"{c['kindok:' + k[5:]]}/{v}" for k, v in c.items() if k.startswith("kind:")})
pickle.dump(errs, open(CACHE / "p2_nlp3_val_errs.pkl", "wb"))
for k, L in errs.items():
    print("=====", k, len(L))
    for e in L[:8]: print("  ", e[0], e[1], "|", e[2][:220], "\n      ĐÚNG", e[3], "\n      ĐỌC ", e[4])
