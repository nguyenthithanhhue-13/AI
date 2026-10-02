"""Các hàm cắt ảnh (crop) dùng chung cho huấn luyện và dự đoán phần CV."""
import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

PAD = 64            # viền đệm để cắt sát mép ảnh
DET_WIN = 24        # cửa sổ bộ dò, trên ảnh thu nhỏ 1/2 (tức 48px ảnh gốc)
EDGE_W, EDGE_H = 48, 24
NODE_CLASSES = ["plain", "robot_UP", "robot_DOWN", "robot_LEFT", "robot_RIGHT",
                "library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
LOOKS = ["none", "normal", "crowded", "covered", "closed"]
ROAD_KINDS = ["normal", "crowded", "covered"]
# ý nghĩa của một dòng chú giải (đọc từ chữ): 10 loại địa điểm + 8 loại khác
LEGEND_KINDS = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate",
                "normal", "crowded", "covered", "closed", "stairs", "oneway", "robot", "weather",
                "none"]      # "none" = không phải dòng chú giải (ký hiệu trên bản đồ, chữ tiêu đề...)
K_NONE = 18
# căn tâm giao lộ: độ lệch (theo đơn vị `unit`) được chia thành các bậc 0.015, từ -6 đến +6 bậc
ALIGN_STEP, ALIGN_BINS = 0.015, 6
ALIGN_MAX = 0.095
DET_CLASSES = ["bg", "node", "landmark", "robot", "swatch", "weather"]


def read_rgb(path):
    """Ảnh PNG lưu dạng bảng màu -> luôn chuyển sang RGB."""
    return np.asarray(Image.open(path).convert("RGB"))


def grid_unit(rc, xy):
    """unit = khoảng cách lưới nhỏ hơn trong hai chiều, từ khớp affine (hàng, cột) -> (x, y)."""
    rc = np.asarray(rc, float)
    xy = np.asarray(xy, float)
    A = np.c_[rc[:, 1], rc[:, 0], np.ones(len(rc))]
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return float(min(np.hypot(*coef[0]), np.hypot(*coef[1])))


def degrade(img, rng):
    """Giả lập ảnh chụp/quét: làm mờ + nén JPEG (dùng để tăng cường dữ liệu train)."""
    out = img
    if rng.random() < 0.8:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.6, 1.0)))
    if rng.random() < 0.8:
        q = int(rng.integers(40, 88))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


class Img:
    """Giữ ảnh gốc (có đệm viền) và ảnh thu nhỏ 1/2 để cắt nhanh."""

    def __init__(self, rgb):
        self.h, self.w = rgb.shape[:2]
        self.full = cv2.copyMakeBorder(rgb, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        half = cv2.resize(rgb, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        self.half = cv2.copyMakeBorder(half, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        quarter = cv2.resize(rgb, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        self.quarter = cv2.copyMakeBorder(quarter, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        self.gray = cv2.cvtColor(self.full, cv2.COLOR_RGB2GRAY)

    def _clip(self, x, y):
        return min(max(x, -PAD // 2), self.w + PAD // 2), min(max(y, -PAD // 2), self.h + PAD // 2)

    def det_patch(self, x, y):
        """Hai cửa sổ 24x24x3 cùng tâm (x, y): chi tiết (ảnh 1/2, phủ 48px) + ngữ cảnh (ảnh 1/4, phủ 96px).
        Ngữ cảnh giúp phân biệt mẫu trong chú giải với ký hiệu trên bản đồ."""
        x, y = self._clip(x, y)
        gx, gy = int(round(x / 4)), int(round(y / 4))     # bám lưới 4px giống lúc quét dày
        r = DET_WIN // 2
        cx, cy = gx * 2 + PAD, gy * 2 + PAD
        a = self.half[cy - r:cy + r, cx - r:cx + r]
        qx, qy = gx + PAD, gy + PAD
        b = self.quarter[qy - r:qy + r, qx - r:qx + r]
        return np.concatenate([a.reshape(-1), b.reshape(-1)])

    def _warp(self, src, cx, cy, ux, uy, span_w, span_h, out_w, out_h):
        """Lấy mẫu hình chữ nhật span_w x span_h (px gốc) tâm (cx, cy), trục dài theo (ux, uy) -> out_w x out_h."""
        nx, ny = -uy, ux
        sw, sh = span_w / out_w, span_h / out_h
        M = np.array([[ux * sw, nx * sh, cx + PAD - span_w / 2 * ux - span_h / 2 * nx],
                      [uy * sw, ny * sh, cy + PAD - span_w / 2 * uy - span_h / 2 * ny]], np.float32)
        interp = cv2.INTER_AREA if max(sw, sh) > 1.3 else cv2.INTER_LINEAR
        if interp == cv2.INTER_AREA:
            # warpAffine không hỗ trợ INTER_AREA: làm mượt nhẹ trước khi thu nhỏ để tránh răng cưa
            k = max(sw, sh)
            x0, y0 = int(cx + PAD - span_w), int(cy + PAD - span_w)
            x0, y0 = max(x0, 0), max(y0, 0)
            sub = src[y0:int(cy + PAD + span_w) + 1, x0:int(cx + PAD + span_w) + 1]
            sub = cv2.GaussianBlur(sub, (0, 0), 0.4 * k)
            M = M.copy(); M[0, 2] -= x0; M[1, 2] -= y0
            return cv2.warpAffine(sub, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(src, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def node_crop(self, x, y, unit):
        """Giao lộ, đã chuẩn hóa theo `unit` = khoảng cách lưới nhỏ nhất (ký hiệu to nhỏ theo unit):
        vùng (1.0 x 0.78) unit -> xám 72x56 (đọc chữ viết tắt / nhãn) + màu 36x28."""
        x, y = self._clip(x, y)
        g = self._warp(self.gray, x, y, 1.0, 0.0, 1.0 * unit, 0.78 * unit, 72, 56)
        c = self._warp(self.full, x, y, 1.0, 0.0, 1.0 * unit, 0.78 * unit, 36, 28)
        return np.concatenate([c.reshape(-1), g.reshape(-1)])

    def edge_crop(self, a, b, unit):
        """Dải (0.6 x 0.3) unit quanh trung điểm đoạn a->b -> 48x24, xoay sao cho a->b là chiều +x."""
        ax, ay = a
        bx, by = b
        d = np.hypot(bx - ax, by - ay) + 1e-6
        ux, uy = (bx - ax) / d, (by - ay) / d
        out = self._warp(self.full, (ax + bx) / 2, (ay + by) / 2, ux, uy, 0.6 * unit, 0.3 * unit, EDGE_W, EDGE_H)
        return out.reshape(-1)

    def label_offset(self, centers):
        """Khoảng cách (tâm mẫu -> mép trái chữ) là hằng số trong một chú giải: lấy trung vị trên mọi dòng cho chắc."""
        return float(np.median([self.label_start(x, y) - x for x, y in centers]))

    def swatch_crop(self, x, y, offset=None):
        """Một dòng chú giải: mẫu ký hiệu 36x24 màu (72x48 px gốc) + phần chữ bên phải 110x14 xám (220x28 px gốc)."""
        x, y = self._clip(x, y)
        cx, cy = int(round(x / 2)) + PAD, int(round(y / 2)) + PAD
        a = self.half[cy - 12:cy + 12, cx - 18:cx + 18].reshape(-1)
        xs = x + (offset if offset is not None else self.label_start(x, y) - x)
        # thêm mẫu ở độ phân giải gốc (64x28 xám): cần để phân biệt nét đơn / nét đôi / nét đứt của kiểu "print" khi ảnh mờ
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        g = self.gray[fy - 14:fy + 14, fx - 32:fx + 32].reshape(-1)
        return np.concatenate([a, self.label_patch(xs, y).reshape(-1), g])

    def label_patch(self, xs, y):
        """Chữ của một dòng chú giải, chuẩn hóa theo CỠ CHỮ (cỡ chữ thay đổi theo từng dòng):
        đo chiều cao phần mực của vài ký tự đầu -> cắt vùng (11.2 x 1.3) lần chiều cao -> 144x16 xám."""
        fx, fy = int(round(xs)) + PAD, int(round(y)) + PAD
        fx = min(max(fx, 20), self.gray.shape[1] - 60)
        reg = self.gray[fy - 20:fy + 21, fx - 1:fx + 300].astype(np.int16)
        bg = np.median(reg[:, :60])
        ink = np.abs(reg - bg) > 45

        def extent(w):
            rows = np.nonzero(ink[:, :max(int(w), 8)].any(1))[0]
            # chỉ lấy khối hàng liên tục chứa tâm (tránh dính dòng trên / dưới)
            if len(rows) == 0:
                return 12, 28
            mid = rows[np.argmin(np.abs(rows - 20))]
            lo = hi = mid
            rs = set(rows.tolist())
            while lo - 1 in rs or lo - 2 in rs:
                lo -= 1 if lo - 1 in rs else 2
            while hi + 1 in rs or hi + 2 in rs:
                hi += 1 if hi + 1 in rs else 2
            return lo, hi

        lo, hi = extent(45)
        h = max(hi - lo + 1, 8)
        lo, hi = extent(3.5 * h)
        h = float(min(max(hi - lo + 1, 8), 30))
        cx = xs - 0.2 * h + 5.6 * h
        cyc = y - 20 + (lo + hi) / 2.0
        return self._warp(self.gray, cx, cyc, 1.0, 0.0, 11.2 * h, 1.3 * h, 144, 16)

    def label_start(self, x, y):
        """Tìm mép trái của chữ trong một dòng chú giải: đi từ mẫu ký hiệu sang phải, qua khoảng trống đầu tiên,
        tới cột đầu tiên có 'mực'. Nhờ vậy chữ luôn nằm cùng vị trí trong ảnh cắt dù mẫu ký hiệu rộng hẹp khác nhau."""
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        band = self.gray[fy - 9:fy + 10, fx + 10:fx + 120].astype(np.int16)
        if band.shape[1] < 20:
            return x + 30
        bg = np.median(band)
        ink = (np.abs(band - bg) > 45).sum(0) >= 1
        gap = 0
        seen_gap = False
        for i, v in enumerate(ink):
            if not v:
                gap += 1
                if gap >= 3:
                    seen_gap = True
            else:
                if seen_gap:
                    return x + 10 + i
                gap = 0
        return x + 30

    def ring_color(self, x, y, r0, r1):
        """Màu trung vị trên vành khuyên bán kính [r0, r1] quanh (x, y): 'màu nền' của một ký hiệu tròn."""
        r = int(r1) + 1
        fx, fy = int(round(x)) + PAD, int(round(y)) + PAD
        sub = self.full[fy - r:fy + r + 1, fx - r:fx + r + 1].astype(np.float32)
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        d = np.hypot(xs, ys)
        m = (d >= r0) & (d <= r1)
        return np.median(sub[m], axis=0)

    def weather_crop(self, x, y):
        """Biểu tượng thời tiết: 40x40 trên ảnh 1/2 (80x80 px gốc)."""
        x, y = self._clip(x, y)
        cx, cy = int(round(x / 2)) + PAD, int(round(y / 2)) + PAD
        return self.half[cy - 20:cy + 20, cx - 20:cx + 20].reshape(-1)

    def dense_windows(self):
        """Mọi cửa sổ bộ dò trên lưới 4px (ảnh gốc). Trả về 2 view (H', W', 24, 24, 3): chi tiết và ngữ cảnh."""
        r = DET_WIN // 2
        hq, wq = self.h // 4, self.w // 4
        region = self.half[PAD - r:PAD + 2 * hq + r, PAD - r:PAD + 2 * wq + r]
        v = np.lib.stride_tricks.sliding_window_view(region, (DET_WIN, DET_WIN), axis=(0, 1))   # (H, W, 3, 24, 24)
        v = v[::2, ::2].transpose(0, 1, 3, 4, 2)[:hq, :wq]
        regq = self.quarter[PAD - r:PAD + hq + r, PAD - r:PAD + wq + r]
        q = np.lib.stride_tricks.sliding_window_view(regq, (DET_WIN, DET_WIN), axis=(0, 1)).transpose(0, 1, 3, 4, 2)[:hq, :wq]
        return v, q, 4
