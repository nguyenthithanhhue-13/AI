import sys, time, cProfile, pstats
sys.path.insert(0, "src")
import numpy as np
from mlp import MLP

X = np.random.randint(0, 255, (20000, 1728), dtype=np.uint8)
Y = np.random.randint(0, 6, 20000)
m = MLP(1728, [256, 64], [6])
for batch in (256, 1024):
    t = time.time()
    pr = cProfile.Profile(); pr.enable()
    m.fit(X, Y, epochs=1, batch=batch, verbose=False)
    pr.disable()
    print("batch", batch, "time", time.time() - t)
    pstats.Stats(pr).sort_stats("cumtime").print_stats(8)
