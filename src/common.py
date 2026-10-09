"""Hằng số và hàm đọc dữ liệu dùng chung."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "delivery_public"
if not DATA.exists():
    DATA = ROOT.parent / "delivery_public"
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# 0=UP 1=DOWN 2=LEFT 3=RIGHT, tính theo (hàng, cột)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DRC = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPP = [1, 0, 3, 2]
RIGHT_OF = [3, 2, 0, 1]   # quay phải 90 độ: UP->RIGHT, DOWN->LEFT, LEFT->UP, RIGHT->DOWN
LEFT_OF = [2, 3, 1, 0]
PLACES = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
STATUS = ["normal", "crowded", "covered", "closed"]


def rel_turn(h, d):
    """Quan hệ giữa hướng đang quay h và hướng đi d: 'S' thẳng, 'R' phải, 'L' trái, 'B' quay đầu."""
    if d == h:
        return "S"
    if d == OPP[h]:
        return "B"
    return "R" if d == RIGHT_OF[h] else "L"


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_split(split):
    d = DATA / split
    rows = load_json(d / "observations.json")
    labels = load_json(d / "labels.json") if (d / "labels.json").exists() else None
    scenes = load_json(d / "scenes.json") if (d / "scenes.json").exists() else None
    return rows, labels, scenes


def macro_accuracy(rows, labels, preds):
    per = defaultdict(list)
    for row, y, p in zip(rows, labels, preds):
        per[row["robot_id"]].append(y == p)
    scores = {r: sum(v) / len(v) for r, v in sorted(per.items())}
    return sum(scores.values()) / len(scores), scores


def world_from_scene(s):
    """Chuyển một phần tử scenes.json (thông tin đúng) thành 'world' mà module chiến thuật dùng.

    world = {
      'adj': {(r,c): {d: (status, stairs, allowed)}}   # allowed=False nếu đi ngược một chiều
      'landmarks': {type: [(r,c), ...]},
      'robot': (r,c), 'heading': int, 'rain': bool }
    Phần CV cũng phải sinh ra đúng cấu trúc này.
    """
    adj = defaultdict(dict)
    for e in s["edges"]:
        a, b = tuple(e["a"]), tuple(e["b"])
        d = DRC.index((b[0] - a[0], b[1] - a[1]))
        ow = tuple(e["oneway_to"]) if e["oneway_to"] else None
        adj[a][d] = (e["status"], e["stairs"], ow is None or ow == b)
        adj[b][OPP[d]] = (e["status"], e["stairs"], ow is None or ow == a)
    lms = defaultdict(list)
    for lm in s["landmarks"]:
        lms[lm["type"]].append(tuple(lm["rc"]))
    return {
        "adj": dict(adj),
        "landmarks": dict(lms),
        "robot": tuple(s["robot"]["rc"]),
        "heading": ACTIONS.index(s["robot"]["heading"]),
        "rain": s["weather"] == "rain",
    }


def mission_from_scene(s):
    m = s["mission"]
    return {
        "goal": m["goal"],
        "goal_ref": (m["goal_ref"]["kind"], m["goal_ref"]["anchor"]) if m["goal_ref"] else None,
        "via": m["via"],
        "via_ref": (m["via_ref"]["kind"], m["via_ref"]["anchor"]) if m["via_ref"] else None,
        "urgent": m["urgent"],
        "fragile": m["fragile"],
    }


def with_strong(X, Y, key, mode="dev", extra=()):
    """Nối thêm bản làm méo MẠNH (cache/cvstrong_train.npz, tạo bằng `cv_data.py train strong`) vào dữ liệu train
    của một CNN. Tắt bằng biến môi trường USE_STRONG=0.
    (đợt 19) MULTI=1: nối thêm bản tăng cường NHIỀU LỚP (cvmulti_train.npz, cvmulti2_train.npz nếu có, và
    cvmulti_validation.npz khi học final). `extra`: các cặp (X, Y) nối cùng lúc (vd validation khi học final).
    Nối MỘT lần vào mảng cấp sẵn, đọc từng file rồi bỏ: không tốn gấp đôi RAM như np.concatenate liên tiếp."""
    import os
    import zipfile
    import numpy as np
    files = []
    if os.environ.get("USE_STRONG", "1") == "1" and (CACHE / "cvstrong_train.npz").exists():
        files.append(CACHE / "cvstrong_train.npz")
    if os.environ.get("MULTI", "0") == "1":
        names = ["cvmulti_train.npz", "cvmulti2_train.npz"] + (["cvmulti_validation.npz"] if mode.startswith("final") else [])
        files += [CACHE / n for n in names if (CACHE / n).exists()]

    def rows_of(f):
        with zipfile.ZipFile(f) as z, z.open(key + "_x.npy") as fh:
            v = np.lib.format.read_magic(fh)
            return np.lib.format._read_array_header(fh, v)[0][0]

    sizes = [rows_of(f) for f in files] + [len(x) for x, _ in extra]
    if not sizes:
        return X, Y
    N = len(X) + sum(sizes)
    Xo = np.empty((N,) + X.shape[1:], X.dtype); Yo = np.empty((N,) + Y.shape[1:], Y.dtype)
    Xo[:len(X)] = X; Yo[:len(X)] = Y
    pos = len(X)
    for f, n in zip(files, sizes):
        d = np.load(f)
        Xo[pos:pos + n] = d[key + "_x"]; Yo[pos:pos + n] = d[key + "_y"]; pos += n
        print(f"   + {n} mẫu từ {f.name} ({key})", flush=True)
        del d
    for x, y in extra:
        Xo[pos:pos + len(x)] = x; Yo[pos:pos + len(x)] = y; pos += len(x)
    return Xo, Yo


def init_from(net, name, mode):
    """(đợt 19) INIT=1: tinh chỉnh tiếp từ trọng số đã học (models_dev cho bản dev*, models_final cho bản final*)
    thay vì học lại từ đầu."""
    import os
    import torch
    if os.environ.get("INIT", "0") != "1":
        return False
    base = OUT / ("models_final" if mode.startswith("final") else "models_dev") / name
    net.load_state_dict(torch.load(base))
    print("   tinh chỉnh tiếp từ", base, flush=True)
    return True
