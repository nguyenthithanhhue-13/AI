import numpy as np, cv2, sys
d = np.load("cache/cvdata_train_100.npz")
rng = np.random.default_rng(0)
# edge crops: mỗi hàng một loại nhãn
rows = []
ey = d["edge_y"]
for name, mask in [("none", ey[:, 0] == 0), ("normal", ey[:, 0] == 1), ("crowded", ey[:, 0] == 2), ("covered", ey[:, 0] == 3),
                   ("closed", ey[:, 0] == 4), ("stairs", ey[:, 1] == 1), ("ow a->b", ey[:, 2] == 1), ("ow b->a", ey[:, 2] == 2)]:
    idx = rng.choice(np.nonzero(mask)[0], 16, replace=False)
    rows.append(np.hstack([d["edge_x"][i].reshape(24, 48, 3) for i in idx]))
E = np.vstack(rows)
E = cv2.resize(E, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("scratch/vis_edges.png", cv2.cvtColor(E, cv2.COLOR_RGB2BGR))
rows = []
for c in range(6):
    idx = rng.choice(np.nonzero(d["det_y"] == c)[0], 30, replace=False)
    rows.append(np.hstack([d["det_x"][i].reshape(24, 24, 3) for i in idx]))
D = cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("scratch/vis_det.png", cv2.cvtColor(D, cv2.COLOR_RGB2BGR))
rows = []
for c in [0, 1, 2, 3, 4, 5, 6, 9, 14]:
    idx = rng.choice(np.nonzero(d["node_y"] == c)[0], 12, replace=False)
    rows.append(np.hstack([d["node_x"][i][3072:].reshape(48, 88) for i in idx]))
cv2.imwrite("scratch/vis_nodes.png", np.vstack(rows))
