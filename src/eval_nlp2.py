"""Đo NLP bản 2 (nlp2.py), có dùng bản đồ ĐÚNG để chốt loại địa điểm lạ (bước resolve_with_map).

    python src/eval_nlp2.py [số ví dụ sai muốn in]

  (a) học train -> đo validation: mô phỏng gặp cách nói hoàn toàn mới (giống tình huống test)
  (b) học train + 4/5 validation -> đo 1/5 còn lại
"""
import sys
from collections import Counter

from common import *
from nlp2 import MissionParser2, resolve_with_map
from strategy import predict

FIELDS = ["goal", "via", "goal_ref", "via_ref", "urgent", "fragile"]


def gold(m):
    return {"goal": m["goal"], "via": m["via"],
            "goal_ref": (m["goal_ref"]["kind"], m["goal_ref"]["anchor"]) if m["goal_ref"] else None,
            "via_ref": (m["via_ref"]["kind"], m["via_ref"]["anchor"]) if m["via_ref"] else None,
            "urgent": m["urgent"], "fragile": m["fragile"]}


def evaluate(parser, scenes, labels_by_scene, show=0, tag=""):
    ok = Counter(); allok = 0; shown = 0; preds = []; ys = []
    for s, y in zip(scenes, labels_by_scene):
        w = world_from_scene(s)
        g = gold(s["mission"])
        raw = parser.parse(s["mission"]["text"])
        p = resolve_with_map(raw, w["landmarks"])
        good = True
        for f in FIELDS:
            ok[f] += p[f] == g[f]; good &= p[f] == g[f]
        allok += good
        pr = [predict(w, p, r) for r in range(10)]
        preds += pr; ys += y
        if pr != y and shown < show:
            shown += 1
            print("   SAI:", s["mission"]["text"], "\n      gold", g, "\n      pred", p,
                  "\n      mentions", [(m_["text"], m_["type"]) for m_ in parser._mentions(s["mission"]["text"])[0]])
    n = len(scenes)
    rows = [{"robot_id": i % 10} for i in range(len(preds))]
    score, per = macro_accuracy(rows, ys, preds)
    print(f"   {tag} " + "  ".join(f"{f}={ok[f] / n:.3f}" for f in FIELDS) + f"  ALL={allok / n:.3f}  | ĐIỂM (bản đồ đúng) = {score:.4f}")
    return score, allok, preds


if __name__ == "__main__":
    show = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    _, trl, trs = load_split("train"); _, val, vas = load_split("validation")
    Yv = [val[i * 10:(i + 1) * 10] for i in range(len(vas))]
    p = MissionParser2().fit([s["mission"] for s in trs], verbose=True)
    print("(0) học train -> đo train (kiểm tra các luật không làm hỏng những gì đã học)")
    evaluate(p, trs, [trl[i * 10:(i + 1) * 10] for i in range(len(trs))], 0)
    print("(a) học train -> đo validation")
    evaluate(p, vas, Yv, show)
    if "--quick" in sys.argv:
        sys.exit()
    print("(b) học train + 4/5 validation -> đo 1/5 còn lại")
    allp = [None] * len(vas)
    tot = 0
    for k in range(5):
        held = [i for i in range(len(vas)) if i % 5 == k]
        rest = [vas[i]["mission"] for i in range(len(vas)) if i % 5 != k]
        pk = MissionParser2().fit([s["mission"] for s in trs] + rest)
        sc, a, pr = evaluate(pk, [vas[i] for i in held], [Yv[i] for i in held], 0, f"fold {k}")
        tot += sc * len(held)
    print(f"   5-fold ĐIỂM (bản đồ đúng) = {tot / len(vas):.4f}")
