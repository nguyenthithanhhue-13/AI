"""Độ chính xác từng robot của cấu hình chiến thuật HIỆN TẠI (bản chỉ học train) với THÔNG TIN ĐÚNG, trên train và validation."""
import sys, json, pickle, collections
sys.path.insert(0, "scratch"); sys.path.insert(0, "src")
from p2_lib import *
from strat_ml import StrategyML, is_night
import vin, hyper, mapref
st = StrategyML.load(OUT / "strategy_ml_trainonly.pkl")
st.hybrid = json.load(open(OUT / "strategy_hybrid.json", encoding="utf-8"))
st.ml4 = pickle.load(open(OUT / "strategy_ml4_trainonly.pkl", "rb")) if (OUT / "strategy_ml4_trainonly.pkl").exists() else {}
st.vin = vin.load(OUT / "vin_trainonly.npz")
st.hyper = hyper.load(OUT / "hyper_trainonly.npz")
st.hypers = {"": st.hyper, "goal": hyper.load(OUT / "hypergoal_trainonly.npz")}
for split in sys.argv[1:] or ["validation"]:
    D = load(split); c = collections.Counter()
    for x in D:
        m = x["m"]; mapg = bool(m["goal_ref"]) and m["goal_ref"]["kind"] in hyper.KINDS_MAP
        p = st.predict_scene(x["w"], x["legs"], m["urgent"], m["fragile"], bool(m["via"]), x["s"]["style"] == "night", mapg)
        for r in range(10):
            c[r] += p[r] == x["y"][r]
    print(split, " ".join(f"R{r}={c[r] / len(D):.3f}" for r in range(10)), f"TB={sum(c.values()) / len(D) / 10:.4f}", flush=True)
