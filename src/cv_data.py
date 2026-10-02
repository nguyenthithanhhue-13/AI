"""Tạo dữ liệu huấn luyện cho các mô hình CV từ scenes.json (chạy song song theo cảnh, lưu cache .npz).

    python src/cv_data.py train            # cache/cvdata_train.npz
    python src/cv_data.py validation
    python src/cv_data.py train 100        # thử nhanh trên 100 cảnh
"""
import sys
import time
from multiprocessing import Pool

import numpy as np

from common import *
from cvfeat import *


def build_scene(args):
    split, s, idx, aug = args
    rng = np.random.default_rng(idx * 7 + (1000003 if aug else 0))
    rgb = read_rgb(DATA / split / s["image"])
    if aug:
        rgb = degrade(rgb, rng)
    im = Img(rgb)
    W, H = im.w, im.h
    xy = {tuple(n["rc"]): tuple(n["xy"]) for n in s["nodes"]}
    lm = {tuple(l["rc"]): l["type"] for l in s["landmarks"]}
    robot = tuple(s["robot"]["rc"])
    out = {k: [] for k in ("det_x", "det_y", "node_x", "node_y", "cn_x", "cn_y", "al_x", "al_y", "edge_x", "edge_y", "sw_x", "sw_y", "we_x", "we_y")}

    # ---- bộ dò (M1) ----
    pos = []   # (x, y, class)
    for rc, p in xy.items():
        c = 3 if rc == robot else (2 if rc in lm else 1)
        pos.append((p[0], p[1], c))
    for l in s["legend"]:
        b = l["swatch"]
        pos.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 4))
    wb = s["weather_box"]
    wc = ((wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2)
    pos.append((wc[0], wc[1], 5))
    P = np.array([(p[0], p[1]) for p in pos])

    def far_enough(x, y, d=13):
        return np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) >= d

    for (x, y, c) in pos:
        if c == 1 and rng.random() < 0.3:
            continue
        dx, dy = rng.integers(-2, 3, 2)
        out["det_x"].append(im.det_patch(x + dx, y + dy)); out["det_y"].append(c)
        if rng.random() < 0.6:   # âm bản sát cạnh: lệch 12-24px so với tâm
            ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(13, 24)
            nx, ny = x + rad * np.cos(ang), y + rad * np.sin(ang)
            if far_enough(nx, ny):
                out["det_x"].append(im.det_patch(nx, ny)); out["det_y"].append(0)
    for _ in range(45):          # âm bản ngẫu nhiên
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if far_enough(x, y):
            out["det_x"].append(im.det_patch(x, y)); out["det_y"].append(0)
    for e in s["edges"]:         # âm bản dọc đoạn đường (ký hiệu bậc thang, mũi tên, dấu X...)
        if rng.random() < 0.25:
            a, b = xy[tuple(e["a"])], xy[tuple(e["b"])]
            t = rng.uniform(0.3, 0.7)
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            if far_enough(x, y):
                out["det_x"].append(im.det_patch(x, y)); out["det_y"].append(0)
    for l in s["legend"]:        # âm bản trên chữ của chú giải
        if rng.random() < 0.5:
            b = l["label"]
            x, y = rng.uniform(b[0], b[2]), (b[1] + b[3]) / 2
            if far_enough(x, y):
                out["det_x"].append(im.det_patch(x, y)); out["det_y"].append(0)

    # ---- giao lộ (M2) ----
    unit = grid_unit(list(xy.keys()), list(xy.values()))
    for rc, p in xy.items():
        reps = 2
        if rc == robot:
            c = 1 + ACTIONS.index(s["robot"]["heading"])
        elif rc in lm:
            c = 5 + PLACES.index(lm[rc])
        else:
            c = 0
            reps = 1 if rng.random() < 0.4 else 0
        for _ in range(reps):       # bộ phân loại: tâm gần đúng (lệch nhỏ), vì trước đó đã có bước căn tâm
            dx, dy = rng.uniform(-0.02, 0.02, 2) * unit
            u = unit * rng.uniform(0.95, 1.05)
            out["node_x"].append(im.node_crop(p[0] + dx, p[1] + dy, u)); out["node_y"].append(c)
        for _ in range(3 if c else (1 if rng.random() < 0.35 else 0)):   # dữ liệu cho CNN: tâm lệch tới 7% unit (như bộ dò thật)
            j = rng.uniform(-0.07, 0.07, 2) * unit
            u = unit * rng.uniform(0.94, 1.06)
            out["cn_x"].append(im.node_crop(p[0] + j[0], p[1] + j[1], u)); out["cn_y"].append(c)
        for _ in range(4 if c else 0):   # bộ căn tâm (chỉ cho robot/địa điểm): lệch lớn, nhãn = độ lệch
            j = rng.uniform(-ALIGN_MAX, ALIGN_MAX, 2)
            u = unit * rng.uniform(0.95, 1.05)
            out["al_x"].append(im.node_crop(p[0] + j[0] * unit, p[1] + j[1] * unit, u)[3024:])
            out["al_y"].append(tuple(np.clip(np.round(j / ALIGN_STEP), -ALIGN_BINS, ALIGN_BINS).astype(int) + ALIGN_BINS))

    # ---- đoạn đường (M3) ----
    look = s["road_look"]
    edges = {(tuple(e["a"]), tuple(e["b"])): e for e in s["edges"]}
    for rc, p in xy.items():
        for d in (1, 3):   # DOWN, RIGHT
            rc2 = (rc[0] + DRC[d][0], rc[1] + DRC[d][1])
            if rc2 not in xy:
                continue
            e = edges.get((rc, rc2))
            if e is None:
                y = (0, 0, 0)
            else:
                lk = 4 if e["status"] == "closed" else LOOKS.index(look[e["status"]])
                ow = 0 if e["oneway_to"] is None else (1 if tuple(e["oneway_to"]) == rc2 else 2)
                y = (lk, int(e["stairs"]), ow)
            ja, jb = rng.uniform(-0.03, 0.03, 2) * unit, rng.uniform(-0.03, 0.03, 2) * unit
            a = (p[0] + ja[0], p[1] + ja[1]); b = (xy[rc2][0] + jb[0], xy[rc2][1] + jb[1])
            out["edge_x"].append(im.edge_crop(a, b, unit * rng.uniform(0.94, 1.06))); out["edge_y"].append(y)

    # ---- chú giải (M4) ----
    centers = [((l["swatch"][0] + l["swatch"][2]) / 2, (l["swatch"][1] + l["swatch"][3]) / 2) for l in s["legend"]]
    loff = im.label_offset(centers)
    for l in s["legend"]:
        b = l["swatch"]
        k = l["kind"][6:] if l["kind"].startswith("place:") else l["kind"]
        kind = LEGEND_KINDS.index(k)                                   # ý nghĩa (đọc từ chữ)
        lk = ROAD_KINDS.index(look[k]) if k in ROAD_KINDS else -1      # KIỂU NÉT của mẫu (chỉ với 3 dòng đường)
        wt = -1
        if k == "weather":                                             # chữ "Thời tiết: mưa / nắng..." (nếu có)
            wt = 1 if "mưa" in l["text"] else (2 if ":" in l["text"] else 0)
        for _ in range(2):
            dx, dy = rng.integers(-4, 5, 2)
            out["sw_x"].append(im.swatch_crop((b[0] + b[2]) / 2 + dx, (b[1] + b[3]) / 2 + dy, loff + rng.integers(-2, 3)))
            out["sw_y"].append((kind, lk, wt))
        dx, dy = rng.integers(-4, 5, 2)     # thêm một bản tự tìm mép chữ (giống lượt đọc đầu tiên lúc dự đoán)
        out["sw_x"].append(im.swatch_crop((b[0] + b[2]) / 2 + dx, (b[1] + b[3]) / 2 + dy)); out["sw_y"].append((kind, lk, wt))
    # âm bản "không phải dòng chú giải": giao lộ / địa điểm / robot trên bản đồ, giữa đoạn đường, biểu tượng thời tiết, điểm ngẫu nhiên
    neg = [xy[rc] for rc in lm] + [xy[robot]]
    plain = [p for rc, p in xy.items() if rc not in lm and rc != robot]
    neg += [plain[i] for i in rng.choice(len(plain), min(6, len(plain)), replace=False)]
    for e in [s["edges"][i] for i in rng.choice(len(s["edges"]), min(5, len(s["edges"])), replace=False)]:
        a, b = xy[tuple(e["a"])], xy[tuple(e["b"])]
        neg.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    neg.append(wc)
    neg += [(rng.uniform(30, W - 30), rng.uniform(10, H - 10)) for _ in range(3)]
    for (x, y) in neg:
        if min(np.hypot(x - c[0], y - c[1]) for c in centers) < 16:
            continue
        dx, dy = rng.integers(-3, 4, 2)
        out["sw_x"].append(im.swatch_crop(x + dx, y + dy)); out["sw_y"].append((K_NONE, -1, -1))

    # ---- thời tiết (M5) ----
    for _ in range(6):
        dx, dy = rng.integers(-7, 8, 2)
        out["we_x"].append(im.weather_crop(wc[0] + dx, wc[1] + dy)); out["we_y"].append(int(s["weather"] == "rain"))
    return {k: np.array(v) for k, v in out.items()}


def build(split, limit=None, aug=True, procs=11):
    scenes = load_split(split)[2]
    if limit:
        scenes = scenes[:limit]
    jobs = [(split, s, i, False) for i, s in enumerate(scenes)]
    if aug:   # thêm một bản "xuống cấp" cho các ảnh sạch
        jobs += [(split, s, i, True) for i, s in enumerate(scenes)
                 if s["degradation"]["blur"] == 0 and s["degradation"]["jpeg_quality"] is None]
    t = time.time()
    with Pool(procs) as p:
        parts = p.map(build_scene, jobs, chunksize=8)
    data = {}
    for k in parts[0]:
        data[k] = np.concatenate([q[k] for q in parts if len(q[k])])
    print(f"{split}: {len(jobs)} ảnh trong {time.time() - t:.0f}s ->", {k: v.shape for k, v in data.items()})
    return data


if __name__ == "__main__":
    split = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else "all"      # all | det | crops | cnn
    data = build(split, None, aug=(split == "train"))
    if what == "cnn":       # chỉ lưu dữ liệu cho CNN giao lộ, không đụng tới các file cache khác
        np.savez(CACHE / f"cvcnn_{split}.npz", cn_x=data["cn_x"], cn_y=data["cn_y"])
        print("đã lưu", CACHE / f"cvcnn_{split}.npz", data["cn_x"].shape)
        sys.exit()
    data = {k: v for k, v in data.items() if not k.startswith("cn_")}
    if what in ("all", "crops"):
        np.savez(CACHE / f"cvdata_{split}.npz", **{k: v for k, v in data.items() if not k.startswith("det")})
    if what in ("all", "det"):
        np.savez(CACHE / f"cvdet_{split}.npz", **{k: v for k, v in data.items() if k.startswith("det")})
    print("đã lưu cache cho", split, what)
