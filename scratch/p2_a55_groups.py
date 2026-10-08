import sys, json, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_fast import *
from strat_ml import StrategyML, group_key
H = json.load(open("outputs/strategy_hybrid.json", encoding="utf-8"))
st = StrategyML({}, H)
r = int(sys.argv[1])
for split in ("train", "validation"):
    c = collections.Counter()
    for x in load(split):
        w = x["w"]; cfg = H[str(r)]
        night = x["s"]["style"] == "night"
        d = st._cost_move(w, x["legs"], r, cfg, night, x["m"]["urgent"], x["m"]["fragile"])
        ok = d == x["y"][r]
        for k, v in (("ghé", bool(x["m"]["via"])), ("đêm", night), ("mưa", w["rain"]), ("gấp", x["m"]["urgent"]), ("dễ vỡ", x["m"]["fragile"])):
            c[(k, v, "n")] += 1; c[(k, v, "ok")] += ok
    print(split, "  ".join(f"{k}={v}: {c[(k,v,'ok')]/c[(k,v,'n')]:.2f}" for k in ("ghé", "đêm", "mưa", "gấp", "dễ vỡ") for v in (True, False)))
