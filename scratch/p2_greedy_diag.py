"""Soi cảnh SAI của luật tham lam gọn R9 (cache/p2_greedy_exact_<key>.json) trong một nhóm.
    python scratch/p2_greedy_diag.py <key> "<nhóm, vd (0, 1, 0)>" [số cảnh]"""
import sys, json, numpy as np
KEY, GRP = sys.argv[1], sys.argv[2]; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
sys.argv = ["x", KEY]
g = {"__name__": "x"}
exec(open("scratch/p2_greedy_exact.py", encoding="utf-8").read().split('\nif __name__ == "__main__":\n')[0], g)
p = json.load(open(f"cache/p2_greedy_exact_{KEY}.json"))[GRP]
X, K, Y = g["prep"]("train"); D = g["load"]("train")
it = [i for i, k in enumerate(K) if str(k) == GRP]
pr = g["score"](X[it], p); n = 0
for j, i in enumerate(it):
    if pr[j] == Y[i]:
        continue
    n += 1
    if n > NS:
        continue
    w = D[i]["w"]
    print("==", D[i]["s"]["scene_id"], "robot", w["robot"], "mũi", "UDLR"[w["heading"]], "điểm đến", D[i]["legs"][0],
          "nhãn", "UDLR"[Y[i]], "đoán", "UDLR"[pr[j]], "(có ghé)" if len(D[i]["legs"]) == 2 else "")
    for d in range(4):
        v = X[i, d]
        if np.isnan(v[0]):
            continue
        rel = "SRLB"[int(np.argmax(v[23:27]))]
        print("    ", "UDLR"[d], rel, "man=%d euc=%.2f bfs=%d px=%.2f" % (v[0], v[1], v[2], v[3]), "đông" if v[8] else "", "mái" if v[9] else "",
              ("lm:" + g["PL"][int(np.argmax(v[13:23]))]) if v[12] else "")
print("sai", n, "/", len(it))
