import sys, pickle, numpy as np
grid, spl, ns, res = pickle.load(open("cache/p2_grid1.pkl", "rb"))
G = grid[:len(res)]
for r in range(10):
    print(f"=== R{r}")
    for si, sn in enumerate(spl):
        ins = res[:, r, si, 0]; uni = res[:, r, si, 1]
        order = np.lexsort((-uni, -ins))[:3]
        print(f"  {sn:6s} n={ns[si]:4d} " + " | ".join(f"{G[i]} ins={ins[i]/ns[si]:.3f} uni={uni[i]/ns[si]:.3f}" for i in order))
