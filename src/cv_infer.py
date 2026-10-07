"""CV: ảnh -> 'world' (đồ thị + địa điểm + robot + thời tiết) mà module chiến thuật dùng. Không dùng scenes.json.

Các bước:
  1. Bộ dò quét dày (MLP trên cửa sổ 48px + ngữ cảnh 96px, bước 4px) -> bản đồ nhiệt:
     giao lộ / địa điểm / robot / dòng chú giải / biểu tượng thời tiết.
  2. Tách vùng chú giải (các mẫu ký hiệu xếp thẳng cột) khỏi vùng bản đồ.
  3. Khớp lưới: ước lượng góc xoay + khoảng cách hàng/cột, gán (hàng, cột) cho từng giao lộ.
  4. Đọc chú giải của CHÍNH ảnh đó: mỗi dòng nghĩa là gì (đọc chữ), mẫu của nó vẽ bằng kiểu nét / màu nào.
  5. Phân loại từng cặp giao lộ kề nhau: kiểu nét (không có đường / 3 kiểu / đóng), bậc thang, một chiều;
     đổi kiểu nét -> trạng thái theo chú giải.
  6. Phân loại từng giao lộ (thường / robot + hướng mũi / 10 loại địa điểm), kết hợp màu của mẫu trong chú giải.
  7. Thời tiết từ biểu tượng (và chữ trong chú giải nếu có).
"""
import itertools
import os
from collections import defaultdict

import cv2
import numpy as np

from common import *
from cvfeat import *
from mlp import MLP


class Models:
    NAMES = ("det", "align", "node", "edge", "swatch", "weather")

    def __init__(self, folder):
        for n in self.NAMES:
            setattr(self, n, MLP.load(folder / f"{n}.npz"))
        # CNN đoạn đường (nếu đã huấn luyện): thay cho MLP `edge` vì sai ít hơn nhiều (bậc thang, một chiều)
        self.edge_cnn = None
        if (folder / "edge_cnn.onnx").exists() and os.environ.get("NO_EDGE_CNN") != "1":
            from cv_edge import EdgeCNN
            self.edge_cnn = EdgeCNN(folder / "edge_cnn.onnx")
        if (folder / "weather_cnn.onnx").exists() and os.environ.get("NO_WEATHER_CNN") != "1":
            from cv_weather import WeatherCNN
            self.weather = WeatherCNN(folder / "weather_cnn.onnx")
        # CNN đọc chú giải (nếu đã huấn luyện): sai ít hơn MLP hàng chục lần ở phần đọc chữ
        if (folder / "swatch_cnn.onnx").exists() and os.environ.get("NO_SWATCH_CNN") != "1":
            from cv_swatch import SwatchCNN
            self.swatch = SwatchCNN(folder / "swatch_cnn.onnx")
        # CNN phân loại giao lộ (nếu đã huấn luyện): thay cho cặp "căn tâm + MLP" vì chịu được tâm lệch
        self.node_cnn = None
        if (folder / "node_cnn.onnx").exists() and os.environ.get("NO_NODE_CNN") != "1":
            from cv_cnn import NodeCNN
            self.node_cnn = NodeCNN(folder / "node_cnn.onnx")

    @property
    def n_params(self):
        return sum(getattr(self, n).n_params for n in self.NAMES)    # (edge đã là CNN nếu có: EdgeCNN.n_params)


def dense_detect(im, model, rows_per_chunk=16):
    v, q, step = im.dense_windows()
    H, W = v.shape[:2]
    out = np.empty((H, W, 6), np.float32)
    n = DET_WIN * DET_WIN * 3
    for r in range(0, H, rows_per_chunk):
        a = v[r:r + rows_per_chunk]
        b = q[r:r + rows_per_chunk]
        x = np.concatenate([a.reshape(-1, n), b.reshape(-1, n)], axis=1)
        out[r:r + rows_per_chunk] = model.predict_proba(x)[0].reshape(a.shape[0], W, 6)
    return out, step


def peaks(score, step, thr=0.5, rad=3):
    """Cực đại địa phương của bản đồ điểm số; tinh chỉnh vị trí bằng trọng tâm 5x5. -> list (x, y, score)."""
    dil = cv2.dilate(score, np.ones((2 * rad + 1, 2 * rad + 1), np.uint8))
    ys, xs = np.nonzero((score >= dil - 1e-7) & (score > thr))
    out = []
    H, W = score.shape
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(y - 2, 0), min(y + 3, H), max(x - 2, 0), min(x + 3, W)
        w = score[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        out.append((float((w * gx).sum() / w.sum() * step), float((w * gy).sum() / w.sum() * step), float(score[y, x])))
    out.sort(key=lambda t: -t[2])
    keep = []          # gộp các đỉnh quá gần nhau
    for p in out:
        if all(abs(p[0] - q[0]) > rad * step or abs(p[1] - q[1]) > rad * step for q in keep):
            keep.append(p)
    return keep


def legend_swatches(sw, map_nodes):
    """Chú giải là một bảng: các mẫu thẳng cột (cách nhau 28-70px) hoặc thẳng hàng (cách 120-420px).
    Nối các mẫu là 'hàng xóm' của nhau rồi giữ NHÓM LIÊN THÔNG LỚN NHẤT; báo nhầm lẻ tẻ trên bản đồ bị loại."""
    n = len(sw)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 1) lõi: các cột chặt (thẳng đứng, cách nhau một bước hàng), ghép các cột ngang hàng nhau (cho phép xoay ~3 độ)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = abs(sw[i][0] - sw[j][0]), abs(sw[i][1] - sw[j][1])
            if dx < 7 + 0.06 * dy and 24 < dy < 75:
                parent[find(i)] = find(j)
    size = defaultdict(int)
    for i in range(n):
        size[find(i)] += 1
    in_col = [size[find(i)] >= 2 for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = abs(sw[i][0] - sw[j][0]), abs(sw[i][1] - sw[j][1])
            if in_col[i] and in_col[j] and dy < 8 + 0.06 * dx and 100 < dx < 650:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        if in_col[i]:
            groups[find(i)].append(i)
    if not groups:
        return []
    core = max(groups.values(), key=len)
    members = list(core)

    def box_of(idx):
        xs = [sw[i][0] for i in idx]; ys = [sw[i][1] for i in idx]
        return (min(xs) - 40, min(ys) - 34, max(xs) + 60, max(ys) + 34)

    def swallowed(box):
        return sum(1 for p in map_nodes if box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])

    # 2) các dòng đã xác nhận còn lại (cột chỉ có 1 dòng, dòng bị hở...): nhận thêm nếu việc nới khung chú giải
    #    không "nuốt" thêm giao lộ nào của bản đồ -> ký hiệu trên bản đồ bị đọc nhầm sẽ không bao giờ được nhận
    base = swallowed(box_of(members))
    rest = [i for i in range(n) if i not in members]
    bx = box_of(members)
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    rest.sort(key=lambda i: abs(sw[i][0] - cx) + abs(sw[i][1] - cy))
    for i in rest:
        if swallowed(box_of(members + [i])) <= base:
            members.append(i)
    return [sw[i] for i in members]


def complete_legend(im, models, rows, theta):
    """Chú giải là một bảng đều: nếu bộ dò bỏ sót một ô (cột c, hàng r) thì thử đọc ngay tại vị trí suy ra từ bảng."""
    if len(rows) < 3:
        return rows
    pts = np.array([(p[0], p[1]) for p in rows])
    c, s = np.cos(np.radians(-theta)), np.sin(np.radians(-theta))
    M = np.array([[c, s], [-s, c]])
    ctr = pts.mean(0)
    q = (pts - ctr) @ M

    def levels(v, gap):
        order = np.argsort(v)
        groups = [[order[0]]]
        for a, b in zip(order[:-1], order[1:]):
            if v[b] - v[a] > gap:
                groups.append([b])
            else:
                groups[-1].append(b)
        return [float(np.median(v[g])) for g in groups]

    xs, ys = levels(q[:, 0], 60), levels(q[:, 1], 16)
    if len(ys) >= 2:      # thêm hàng bị sót hoàn toàn nằm giữa hai hàng đã thấy
        pitch = float(np.min(np.diff(ys)))
        full = [ys[0]]
        for y in ys[1:]:
            k = int(round((y - full[-1]) / pitch))
            for t in range(1, k):
                full.append(full[-1] + (y - full[-1]) / (k - t + 1))
            full.append(y)
        ys = full
    cands = []
    for x in xs:
        for y in ys:
            if np.min(np.hypot(q[:, 0] - x, q[:, 1] - y)) > 18:
                cands.append(np.array([x, y]) @ M.T + ctr)
    out = list(rows)
    if cands:
        offs = [(0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)]
        X = np.array([im.swatch_crop(p[0] + dx, p[1] + dy) for p in cands for dx, dy in offs])
        pn = models.swatch.predict_proba(X)[0][:, K_NONE].reshape(len(cands), len(offs))
        for p, row in zip(cands, pn):
            k = int(np.argmin(row))
            if row[k] < 0.3 and 10 < p[0] < im.w - 10 and 10 < p[1] < im.h - 10:
                out.append((float(p[0] + offs[k][0]), float(p[1] + offs[k][1]), float(1 - row[k])))
    return out


def find_legend_rows(im, models, det):
    """Mọi đỉnh của bộ dò (dù được coi là giao lộ hay mẫu chú giải) đều được 'đọc thử' như một dòng chú giải:
    dòng thật có chữ bên phải nên mô hình đọc ra được ý nghĩa; ký hiệu trên bản đồ thì rơi vào lớp 'none'."""
    cand = []
    for p in sorted(det["nodes"] + det["swatches"], key=lambda t: -t[2]):
        if all(abs(p[0] - q[0]) > 14 or abs(p[1] - q[1]) > 14 for q in cand):
            cand.append(p)
    if not cand:
        return [], None
    PK = models.swatch.predict_proba(np.array([im.swatch_crop(p[0], p[1]) for p in cand]))[0]
    conf = [p for p, pk in zip(cand, PK) if pk[K_NONE] < 0.5]
    # "giao lộ của bản đồ" = đỉnh giao lộ mạnh mà không được đọc ra là dòng chú giải
    map_nodes = [p for p, pk in zip(cand, PK) if pk[K_NONE] >= 0.5 and p in det["nodes"] and p[2] > 0.5]
    rows = legend_swatches(conf, map_nodes)
    # dòng "Thời tiết" (nếu có): biểu tượng thời tiết nằm ngay trong chú giải
    weather_xy, best = None, 0.5
    for p, pk in zip(cand, PK):
        if p in rows and pk[17] > best:
            weather_xy, best = (p[0], p[1]), pk[17]
    return rows, weather_xy


def fit_axis(vals, spacing):
    """Gom các giá trị 1 chiều thành hàng (hoặc cột) của lưới. -> chỉ số nguyên cho từng giá trị."""
    order = np.argsort(vals)
    clusters = [[order[0]]]
    for a, b in zip(order[:-1], order[1:]):
        if vals[b] - vals[a] > 0.42 * spacing:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    centers = np.array([np.median(vals[c]) for c in clusters])
    if len(centers) >= 2:
        gaps = np.diff(centers)
        ok = gaps[(gaps > 0.6 * spacing) & (gaps < 1.5 * spacing)]
        if len(ok):
            spacing = float(np.median(ok))
    # đi từ cụm đông nhất ra hai phía; cụm nhỏ nằm "lưng chừng" giữa hai hàng/cột (cách cụm đã nhận không phải
    # một số nguyên lần khoảng cách lưới) là báo nhầm (vd: vạch bậc thang bị tưởng là giao lộ) -> loại (-1)
    big = int(np.argmax([len(c) for c in clusters]))
    kint = {big: 0}
    for direction in (1, -1):
        last = big
        ci = big + direction
        while 0 <= ci < len(clusters):
            step = abs(centers[ci] - centers[last]) / spacing
            k = int(round(step))
            if (k == 0 or abs(step - k) > 0.35) and len(clusters[ci]) <= 2:
                kint[ci] = None
            else:
                kint[ci] = kint[last] + direction * max(1, k)
                last = ci
            ci += direction
    kmin = min(v for v in kint.values() if v is not None)
    idx = np.full(len(vals), -1)
    for ci, c in enumerate(clusters):
        if kint[ci] is not None:
            idx[c] = kint[ci] - kmin
    return idx, spacing


def fit_lattice(pts):
    """pts: (n, 2) tọa độ giao lộ. -> (hàng, cột) cho từng điểm, góc xoay (độ), (khoảng cách cột, hàng)."""
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, 1e9)
    d0 = np.median(d.min(1))
    ii, jj = np.nonzero((d > 0.6 * d0) & (d < 1.5 * d0))
    vx, vy = pts[jj, 0] - pts[ii, 0], pts[jj, 1] - pts[ii, 1]
    ang = np.degrees(np.arctan2(vy, vx))
    folded = (ang + 45) % 90 - 45          # gập về [-45, 45): cạnh ngang và dọc đều cho cùng góc xoay
    folded = folded[np.abs(folded) < 15]
    theta = float(np.median(folded)) if len(folded) else 0.0
    c, s = np.cos(np.radians(-theta)), np.sin(np.radians(-theta))
    q = (pts - pts.mean(0)) @ np.array([[c, s], [-s, c]])
    qx, qy = q[jj, 0] - q[ii, 0], q[jj, 1] - q[ii, 1]
    hor = np.abs(qx[np.abs(qy) < 0.3 * np.abs(qx)])
    ver = np.abs(qy[np.abs(qx) < 0.3 * np.abs(qy)])
    sx = float(np.median(hor)) if len(hor) else d0
    sy = float(np.median(ver)) if len(ver) else d0
    cols, sx = fit_axis(q[:, 0], sx)
    rows, sy = fit_axis(q[:, 1], sy)
    return rows, cols, theta, (sx, sy)


def read_legend(im, models, sw, theta):
    """Đọc chú giải. Trả về dict:
       look2status: kiểu nét -> trạng thái (thường/đông/mái che)
       types: tập loại địa điểm có trong chú giải;  place_rows: [(màu mẫu, xác suất loại)]
       rain_text: True/False/None theo chữ "Thời tiết: ..." nếu có."""
    out = {"look2status": {k: k for k in ROAD_KINDS}, "conf": 0.0, "how": "mặc định", "types": None, "place_rows": [],
           "rain_text": None, "agree": None}
    if len(sw) < 4:
        return out
    loff = im.label_offset([(p[0], p[1]) for p in sw])
    PK, PL, PW = models.swatch.predict_proba(np.array([im.swatch_crop(p[0], p[1], loff) for p in sw]))
    kinds = PK.argmax(1)
    # --- thứ tự đọc: theo cột, trong cột từ trên xuống ---
    pts = np.array([(p[0], p[1]) for p in sw])
    c, s = np.cos(np.radians(-theta)), np.sin(np.radians(-theta))
    q = (pts - pts.mean(0)) @ np.array([[c, s], [-s, c]])
    order = np.argsort(q[:, 0])
    col = np.zeros(len(sw), int)
    k = 0
    for a, b in zip(order[:-1], order[1:]):
        if q[b, 0] - q[a, 0] > 60:
            k += 1
        col[b] = k
    reading = sorted(range(len(sw)), key=lambda i: (col[i], q[i, 1]))

    def best_perm(rows):
        """rows: {chỉ số ý nghĩa (0 thường, 1 đông, 2 mái che): chỉ số dòng}. Tìm hoán vị kiểu nét hợp lý nhất;
        nếu chỉ đọc được 2 trong 3 dòng thì kiểu nét còn lại thuộc về ý nghĩa còn lại."""
        best, bp, second = None, -1.0, -1.0
        for perm in itertools.permutations(range(3)):
            p = float(np.prod([PL[r, perm[t]] for t, r in rows.items()]))
            if p > bp:
                best, bp, second = perm, p, bp
            elif p > second:
                second = p
        return {ROAD_KINDS[best[t]]: ROAD_KINDS[t] for t in range(3)}, bp / (bp + second + 1e-9)

    # cách 1: đọc chữ -> dòng nào là "đường thường", "đông người", "có mái che"
    rows_text = None
    cand = {}
    for kk in range(3):
        i = int(np.argmax(PK[:, 10 + kk]))
        if PK[i, 10 + kk] > 0.5:
            cand[kk] = i
    if len(cand) >= 2 and len(set(cand.values())) == len(cand):
        rows_text = cand
    # cách 2: theo bố cục -> ba dòng ngay trước dòng "đường đóng"
    rows_order = None
    pos_closed = [t for t, i in enumerate(reading) if kinds[i] == LEGEND_KINDS.index("closed")]
    if pos_closed and pos_closed[0] >= 3:
        rows_order = {t: reading[pos_closed[0] - 3 + t] for t in range(3)}
    # cách 3 (ưu tiên): kết hợp chữ + bố cục: ba dòng LIỀN NHAU theo thứ tự đọc mà chữ khớp nhất với
    # "thường, đông, mái che" (trong dữ liệu có nhãn, ba dòng này luôn đứng liền nhau theo đúng thứ tự đó)
    rows_joint, best_s = None, 0.0
    for t in range(len(reading) - 2):
        pr = [PK[reading[t + kk], 10 + kk] for kk in range(3)]
        if sum(p > 0.5 for p in pr) < 2:        # cho phép đọc sai chữ ở nhiều nhất một trong ba dòng
            continue
        sc = float(np.prod(np.maximum(pr, 0.03)))
        if sc > best_s:
            rows_joint, best_s = {kk: reading[t + kk] for kk in range(3)}, sc
    if rows_joint is not None:
        out["look2status"], out["conf"] = best_perm(rows_joint)
        out["how"] = "chữ + bố cục"
        out["agree"] = rows_order == rows_joint if rows_order is not None else None
    elif rows_text is not None:
        out["look2status"], out["conf"] = best_perm(rows_text)
        out["how"] = "đọc chữ" if len(rows_text) == 3 else "đọc chữ (2/3 dòng)"
        out["agree"] = rows_order == rows_text if rows_order is not None else None
    elif rows_order is not None:
        out["look2status"], out["conf"] = best_perm(rows_order)
        out["how"] = "theo bố cục"
    # --- các dòng địa điểm ---
    types = set()
    for i in range(len(sw)):
        if kinds[i] < 10 and PK[i, kinds[i]] > 0.5:
            types.add(PLACES[kinds[i]])
            out["place_rows"].append((im.ring_color(sw[i][0], sw[i][1], 9, 12), PK[i, :10].copy()))
    if len(types) >= 3:
        out["types"] = types
    # --- chữ thời tiết ---
    wi = [i for i in range(len(sw)) if kinds[i] == LEGEND_KINDS.index("weather")]
    if wi:
        i = max(wi, key=lambda i: PK[i, 17])
        if PK[i, 17] > 0.5:
            out["weather_xy"] = (sw[i][0], sw[i][1])      # biểu tượng thời tiết nằm ngay trong dòng này
        if PW[i, 1] > 0.9:
            out["rain_text"] = True
        elif PW[i, 2] > 0.9:
            out["rain_text"] = False
    return out


def detect(im, models):
    """Bước tốn thời gian nhất (quét dày) -> chỉ giữ lại danh sách đỉnh để cache."""
    heat, step = dense_detect(im, models.det)
    return {"nodes": peaks(heat[..., 1] + heat[..., 2] + heat[..., 3], step, thr=0.3, rad=3),
            "swatches": peaks(heat[..., 4], step, thr=0.3, rad=2),
            "weather": peaks(heat[..., 5], step, thr=0.2, rad=4)}


WEATHER_FIX = True


def _median_nn(pts):
    """Khoảng cách tới láng giềng gần nhất (trung vị) của một tập đỉnh (x, y, score)."""
    if len(pts) < 3:
        return 100.0
    a = np.array([(p[0], p[1]) for p in pts])
    d = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
    np.fill_diagonal(d, 1e9)
    return float(np.median(d.min(1)))


NODE_FILTER = True
BG_THR = 0.5
GRID_FILL = True       # lấp ô trống của lưới bằng CNN giao lộ
GRID_FILL_THR = 0.5
COLOR_FLOOR = 0.02     # trước 1e-4
EDGE_CNN_RARE = True   # đoạn MLP đọc là đóng / một chiều / bậc thang: luôn hỏi lại CNN
# (kiểu nét, bậc thang, một chiều). CNN đoạn đường học thêm ảnh làm méo mạnh (cv_data.py train strong): validation ảnh gốc
# 300/300, ảnh mờ 1.0 + JPEG 40 chỉ 1 lỗi bậc thang (CNN cũ: 20 + 6 một chiều) -> lấy thẳng CNN, không trung bình với MLP
EDGE_HEAD_MODE = os.environ.get("EDGE_HEAD_MODE", "cnn,cnn,cnn").split(",")
EDGE_CNN_TH = 0.999   # MLP chắc chắn hơn mức này thì tin luôn; còn lại hỏi CNN đoạn đường
# Trọng số của CNN khi ghép với MLP ở bước phân loại giao lộ (1 = chỉ CNN). CNN đúng gần như mọi ca mà MLP đọc sai,
# nên nó phải nặng hơn hẳn; nhưng bỏ hẳn MLP lại tệ đi (train 0,9963), nên giữ một phần nhỏ.
# Đo trên train (2.000 cảnh): 0,5 -> 0,9977 · 0,8 -> 0,9999 · 0,85 -> 0,9999 · 1,0 -> 0,9963; validation 1,0000 với 0,8-0,9.
NODE_CNN_W = 0.8
NODE_TTA = False      # (đã thử: đọc mỗi giao lộ ở vài vị trí / cỡ lệch rồi lấy trung bình -> kết quả y hệt, chậm 4 lần)


def drop_nonnodes(im, models, nodes, thr=BG_THR):
    """Loại các đỉnh dò mà CNN giao lộ (lớp 15) cho là KHÔNG phải giao lộ: chữ tiêu đề, vạch bậc thang, mũi tên...
    Một đỉnh báo nhầm nằm lọt vào lưới làm lệch chỉ số hàng/cột của cả bản đồ."""
    cnn = models.node_cnn
    if not NODE_FILTER or cnn is None or cnn.n_classes <= 15 or len(nodes) < 6:
        return nodes, []
    unit0 = _median_nn(nodes)
    pn = cnn.p_not_node(np.array([im.node_crop(p[0], p[1], unit0) for p in nodes]))
    keep = [p for p, q in zip(nodes, pn) if q < thr]
    if len(keep) < max(6, 0.6 * len(nodes)):       # lọc quá tay -> bỏ qua, giữ nguyên
        return nodes, []
    return keep, [p for p, q in zip(nodes, pn) if q >= thr]


def _in_box(sw, pt, mx=40, my=34):
    sx = [p[0] for p in sw]; sy = [p[1] for p in sw]
    return min(sx) - mx <= pt[0] <= max(sx) + 60 and min(sy) - my <= pt[1] <= max(sy) + my


def analyze(rgb, models, det=None):
    im = Img(rgb)
    if det is None:
        det = detect(im, models)
    all_nodes = [p for p in det["nodes"] if p[2] > 0.5]
    nodes = all_nodes
    wpk = det["weather"]
    wpk_all = det["weather"]
    sw, weather_row = find_legend_rows(im, models, det)
    info = {}
    if sw:      # vùng chú giải: loại các "giao lộ" nằm trong đó
        sx = np.array([p[0] for p in sw]); sy = np.array([p[1] for p in sw])
        box = (sx.min() - 40, sy.min() - 34, sx.max() + 60, sy.max() + 34)
        inside = lambda p: box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]
        kept = [p for p in nodes if not inside(p)]
        if len(kept) >= max(6, 0.5 * len(nodes)):
            nodes = kept
            wpk = [p for p in wpk if not inside(p)]
        else:   # "chú giải" nuốt mất bản đồ -> nhóm mẫu sai; chỉ loại các điểm trùng với mẫu
            nodes = [p for p in nodes if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 24 for q in sw)]
    # biểu tượng thời tiết: nằm ngay trong dòng "Thời tiết" của chú giải, HOẶC ở một góc ảnh
    if weather_row is not None:
        wx, wy = weather_row
    else:
        # Biểu tượng thời tiết chỉ có hai chỗ: một dòng trong chú giải, hoặc góc trên bên phải ảnh (x/W 0.9-0.97, y/H 0.02-0.09
        # trên 100% ảnh train + validation). Đĩa vàng của robot trông giống mặt trời nên bộ dò hay gán "thời tiết" cho robot:
        # vì vậy chỉ nhận đỉnh nằm ở góc hoặc trong khung chú giải.
        def corner(p):
            return p[0] > 0.84 * im.w and p[1] < 0.14 * im.h
        elig = [p for p in wpk_all if p[2] > 0.3 and (corner(p) or (sw and _in_box(sw, (p[0], p[1]))))] if WEATHER_FIX else wpk
        if elig:
            wx, wy, wscore = max(elig, key=lambda t: t[2])
            if wscore > 0.5 and not (WEATHER_FIX and sw and _in_box(sw, (wx, wy))):
                nodes = [p for p in nodes if np.hypot(p[0] - wx, p[1] - wy) > 45]
        else:
            wx, wy = (0.935 * im.w, 0.055 * im.h) if WEATHER_FIX else (im.w - 70, 50)

    # đỉnh yếu (điểm 0.3-0.5) của bộ dò: chỉ dùng để lấp các ô lưới còn trống (vd: nhãn chữ bị xoay nghiêng)
    weak = [p for p in det["nodes"] if 0.3 <= p[2] <= 0.5
            and not (sw and inside(p)) and all(np.hypot(p[0] - q[0], p[1] - q[1]) > 30 for q in nodes)]
    nodes, dropped = drop_nonnodes(im, models, nodes)
    weak = [p for p in weak if p not in dropped]
    legend = None
    filled = False
    for attempt in range(5):
        pts = np.array([(p[0], p[1]) for p in nodes])
        rows, cols, theta, spacing = fit_lattice(pts)
        cell = {}       # mỗi ô lưới giữ một điểm (điểm số cao nhất)
        for i, p in enumerate(nodes):
            if rows[i] < 0 or cols[i] < 0:
                continue
            rc = (int(rows[i]), int(cols[i]))
            if rc not in cell or p[2] > cell[rc][2]:
                cell[rc] = p
        if attempt == 0 and weak and len(cell) >= 6:
            # khớp affine (hàng, cột) -> (x, y); đỉnh yếu nào rơi đúng vào một ô trống của lưới thì nhận thêm
            RC = np.array([[rc[1], rc[0], 1.0] for rc in cell]); XY = np.array([(p[0], p[1]) for p in cell.values()])
            coef, *_ = np.linalg.lstsq(RC, XY, rcond=None)
            A = coef[:2].T
            added = []
            for p in weak:
                cr = np.linalg.solve(A, np.array([p[0], p[1]]) - coef[2])
                c_i, r_i = int(round(cr[0])), int(round(cr[1]))
                if (r_i, c_i) in cell or max(abs(cr[0] - c_i), abs(cr[1] - r_i)) > 0.25:
                    continue
                if not (min(r for r, _ in cell) <= r_i <= max(r for r, _ in cell) and min(c for _, c in cell) <= c_i <= max(c for _, c in cell)):
                    continue
                added.append(p)
            if added:
                nodes = nodes + added
                weak = []
                continue
        if GRID_FILL and not filled and len(cell) >= 6 and models.node_cnn is not None and models.node_cnn.n_classes > 15:
            # Ô trống bên trong lưới: hỏi thẳng CNN giao lộ tại vị trí lưới dự đoán. Bộ dò có thể bỏ sót hẳn một giao lộ
            # (nhãn địa điểm cạnh dấu X, ảnh nén JPEG mạnh). Trên 1.567 ô trống thật (train + validation, cả khi nén JPEG 35)
            # CNN cho p(không phải giao lộ) >= 0.935, còn 95% giao lộ thật < 0.5 (scratch/h25_gridfill_probe.py).
            filled = True
            RC = np.array([[rc[1], rc[0], 1.0] for rc in cell]); XY = np.array([(p[0], p[1]) for p in cell.values()])
            coef, *_ = np.linalg.lstsq(RC, XY, rcond=None)
            u = float(np.median([np.hypot(*coef[0]), np.hypot(*coef[1])]))
            r0, r1 = min(r for r, _ in cell), max(r for r, _ in cell)
            c0, c1 = min(c for _, c in cell), max(c for _, c in cell)
            empty = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in cell]
            if empty:
                P = np.array([[c, r, 1.0] for r, c in empty]) @ coef
                ok_pos = [k for k, (x, y) in enumerate(P) if 0 <= x < im.w and 0 <= y < im.h and not (sw and inside((x, y)))]
                if ok_pos:
                    pn = models.node_cnn.p_not_node(np.array([im.node_crop(P[k][0], P[k][1], u) for k in ok_pos]))
                    add = [(float(P[k][0]), float(P[k][1]), 0.5) for k, q in zip(ok_pos, pn) if q < GRID_FILL_THR]
                    if add:
                        nodes = nodes + add
                        continue
        rcs = sorted(cell)
        xy = {rc: (cell[rc][0], cell[rc][1]) for rc in rcs}
        unit = grid_unit(rcs, [xy[rc] for rc in rcs]) if len(rcs) >= 4 else float(min(spacing))
        if legend is None:
            sw = complete_legend(im, models, sw, theta)
            legend = read_legend(im, models, sw, theta)
        look2status = legend["look2status"]
        # --- đoạn đường ---
        pairs = []
        for rc in rcs:
            for d in (1, 3):
                rc2 = (rc[0] + DRC[d][0], rc[1] + DRC[d][1])
                if rc2 in xy:
                    pairs.append((rc, rc2, d))
        adj = defaultdict(dict)
        if pairs:
            crops = np.array([im.edge_crop(xy[a], xy[b], unit) for a, b, d in pairs])
            EP = models.edge.predict_proba(crops)
            if models.edge_cnn is not None:
                # CNN đọc đoạn đường chính xác hơn nhiều (0 lỗi / 26.738 mảnh validation so với 1 / 10 / 16 của MLP) nhưng
                # chậm hơn ~50 lần trên CPU: chỉ dùng nó cho những đoạn mà MLP KHÔNG chắc (thường dưới 2% số đoạn).
                need = np.zeros(len(pairs), bool)
                for head in EP:
                    need |= head.max(1) < EDGE_CNN_TH
                if EDGE_CNN_RARE:
                    # kết luận hiếm mà hệ trọng (đóng / một chiều / bậc thang) luôn được CNN đọc lại: MLP có khi sai mà rất
                    # tự tin (val cảnh 75, kiểu print bị mờ: đoạn thường -> "đóng" với xác suất 0.999)
                    need |= (EP[0].argmax(1) == LOOKS.index("closed")) | (EP[1][:, 1] > 0.5) | (EP[2].argmax(1) != 2)
                if need.any():
                    CP = models.edge_cnn.predict_proba(crops[need])
                    for k, head in enumerate(EP):
                        # theo từng đầu ra: "cnn" = thay bằng CNN, "avg" = trung bình MLP và CNN.
                        # CNN đọc kiểu nét tốt hơn hẳn, nhưng với ảnh mờ (sigma 1) nó đọc bậc thang / một chiều kém MLP
                        # (scratch/h24_cv_stress.py blur=1.0 jpeg=40)
                        if EDGE_HEAD_MODE[k] == "avg":
                            head[need] = 0.5 * (head[need] + CP[k])
                        else:
                            head[need] = CP[k]
            for (a, b, d), pl, ps, po in zip(pairs, *EP):
                lk = LOOKS[int(np.argmax(pl))]
                if lk == "none":
                    continue
                status = "closed" if lk == "closed" else look2status[lk]
                stairs = bool(ps[1] > 0.5)
                ow = int(np.argmax(po))
                adj[a][d] = (status, stairs, ow != 2)
                adj[b][OPP[d]] = (status, stairs, ow != 1)
        # giao lộ thật luôn có ít nhất một đoạn đường: điểm cô lập là báo nhầm (chữ tiêu đề, góc khung...) -> bỏ, khớp lại lưới
        lonely = [rc for rc in rcs if rc not in adj]
        if not lonely or len(rcs) - len(lonely) < 4:
            break
        drop = {cell[rc] for rc in lonely}
        nodes = [p for p in nodes if p not in drop]
    rcs = [rc for rc in rcs if rc in adj] or rcs

    # --- căn tâm rồi phân loại giao lộ ---
    # bộ dò có thể lệch vài pixel (nhất là với nhãn chữ rộng); mô hình 'align' đoán độ lệch để dời tâm về đúng chỗ
    # (chỉ căn tâm cho robot / địa điểm; giao lộ thường thì giữ nguyên vì căn tâm làm chúng tệ đi)
    crops = np.array([im.node_crop(xy[rc][0], xy[rc][1], unit) for rc in rcs])
    NP = models.node.predict_proba(crops)[0]
    NPc = models.node_cnn.predict_proba(crops) if models.node_cnn is not None else None
    if NPc is not None and NODE_TTA:
        # đọc thêm ở vài vị trí lệch nhẹ và vài cỡ khác rồi lấy trung bình: bù cho việc tâm của bộ dò không chính xác
        for dx, dy, su in ((-0.05, 0, 1.0), (0.05, 0, 1.0), (0, -0.05, 1.0), (0, 0.05, 1.0), (0, 0, 0.93), (0, 0, 1.07)):
            NPc = NPc + models.node_cnn.predict_proba(
                np.array([im.node_crop(xy[rc][0] + dx * unit, xy[rc][1] + dy * unit, unit * su) for rc in rcs]))
        NPc /= NPc.sum(1, keepdims=True)
    todo = [i for i in range(len(rcs)) if NP[i, 0] < 0.9 or (NPc is not None and NPc[i, 0] < 0.9)]
    if todo:
        cxy = {i: xy[rcs[i]] for i in todo}
        bins = np.arange(-ALIGN_BINS, ALIGN_BINS + 1) * ALIGN_STEP * unit
        for _ in range(3):
            A = models.align.predict_proba(np.array([im.node_crop(cxy[i][0], cxy[i][1], unit)[3024:] for i in todo]))
            for i, px, py in zip(todo, A[0], A[1]):
                cxy[i] = (cxy[i][0] - float(bins[px.argmax()]), cxy[i][1] - float(bins[py.argmax()]))
        crops2 = np.array([im.node_crop(cxy[i][0], cxy[i][1], unit) for i in todo])
        NP2 = models.node.predict_proba(crops2)[0]
        xy = dict(xy)
        for i, p in zip(todo, NP2):
            NP[i] = p
            xy[rcs[i]] = cxy[i]            # tâm đã căn: dùng cho việc so màu với mẫu trong chú giải
        if NPc is not None:
            # CNN đọc thêm một lần tại tâm ĐÃ CĂN rồi lấy lần đọc tự tin hơn: tâm của bộ dò có khi lệch 10-20px
            # (nhãn chữ rộng của kiểu "print"), vượt quá biên độ lệch mà CNN được huấn luyện.
            NPc2 = models.node_cnn.predict_proba(crops2)
            for k, i in enumerate(todo):
                if NPc2[k].max() > NPc[i].max():
                    NPc[i] = NPc2[k]
    if NPc is not None:
        # CNN chịu được tâm lệch nên đọc tại vị trí bộ dò; ghép với MLP (đọc tại tâm đã căn) bằng trung bình hình học
        mode = os.environ.get("NODE_COMBINE", "both")
        # CNN sai ít hơn MLP hàng chục lần nên được cân nặng hơn trong trung bình hình học có trọng số
        wc = float(os.environ.get("NODE_CNN_W", NODE_CNN_W))
        NP = NPc if mode == "cnn" else (NPc + 1e-6) ** wc * (NP + 1e-6) ** (1 - wc)
        NP = NP / NP.sum(1, keepdims=True)
    ri = int(np.argmax(NP[:, 1:5].sum(1)))
    robot = rcs[ri]
    heading = int(np.argmax(NP[ri, 1:5]))
    # màu mẫu trong chú giải có phân biệt được các loại không? (kiểu sketch/classic: có; night/print: không)
    # mẫu "có màu" (độ bão hòa cao) thì màu mới mang thông tin; mẫu trắng/đen/xám thì bỏ qua
    rows_c = legend["place_rows"]
    use_color = False
    if len(rows_c) >= 2:
        cols_c = np.array([c for c, _ in rows_c])
        dd = np.abs(cols_c[:, None] - cols_c[None]).sum(-1)[np.triu_indices(len(cols_c), 1)]
        # màu chỉ có ích khi mẫu "có màu" VÀ các loại có màu khác nhau (kiểu night: mọi mẫu cùng một màu nền -> bỏ)
        use_color = np.median(cols_c.max(1) - cols_c.min(1)) > 20 and np.median(dd) > 30
    landmarks = defaultdict(list)
    for i, rc in enumerate(rcs):
        if i == ri:
            continue
        p_land = NP[i, 5:].sum()
        if p_land <= NP[i, 0]:
            continue
        score = np.log(NP[i, 5:] + 1e-6)
        if use_color:
            c = im.ring_color(xy[rc][0], xy[rc][1], 0.115 * unit, 0.15 * unit)
            pc = np.zeros(10)
            for col_r, pk in rows_c:
                pc += pk * np.exp(-np.abs(c - col_r).sum() / 12.0)
            # sàn COLOR_FLOOR: màu mẫu trong chú giải đọc lệch (ô màu nhỏ bị JPEG làm loang -> gần trắng) không được
            # lật kết quả mà CNN đọc chữ rất chắc (val cảnh 190 khi nén JPEG 35: office -> lecture)
            score += np.log(pc / (pc.sum() + 1e-12) + COLOR_FLOOR)
        if legend["types"] is not None:
            for t in range(10):
                if PLACES[t] not in legend["types"]:
                    score[t] -= 6.0
        landmarks[PLACES[int(np.argmax(score))]].append(rc)

    p_rain = float(models.weather.predict_proba(im.weather_crop(wx, wy)[None])[0][0, 1])
    # Thời tiết chỉ lấy từ biểu tượng. (Chữ "Thời tiết: mưa" trong chú giải từng được dùng để ghi đè, nhưng trên validation
    # nó không sửa được ca nào (48/48 ca chữ và biểu tượng đều đúng), mà lại thêm rủi ro khi gặp cách ghi mới.)
    rain = p_rain > 0.5
    world = {"adj": {a: v for a, v in adj.items()}, "landmarks": {k: sorted(v) for k, v in landmarks.items()}, "robot": robot,
             "heading": heading, "rain": bool(rain)}
    info.update({"xy": {rc: xy[rc] for rc in rcs}, "theta": theta, "unit": unit, "look2status": look2status,
                 "legend_conf": legend["conf"], "legend_how": legend["how"], "legend_agree": legend["agree"],
                 "legend_types": legend["types"], "n_swatches": len(sw), "p_rain_icon": p_rain, "rain_text": legend["rain_text"],
                 "use_color": bool(use_color)})
    return world, info
