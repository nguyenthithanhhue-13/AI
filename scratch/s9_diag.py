from s6_diag import *

# R4: mưa có ảnh hưởng tới bậc thang?
for rain in (True, False):
    diag(4, {"crowd": 1}, cond=lambda i: W[i]["rain"] == rain)
    diag(0, {"crowd": 1}, cond=lambda i: W[i]["rain"] == rain)   # legged=False -> không đi bậc thang
print("R4 rain, model = no stairs:")
n = ok = 0
for i in range(len(Y)):
    if W[i]["rain"]:
        q = path_q(W[i], legs_for(W[i], M[i]), {"crowd": 1}, False)
        n += 1; ok += pick(q, W[i]["heading"]) == Y[i][4]
print(ok / n, n)
diag(5, {"turn": 3, "turnB": 20}, show=8)
