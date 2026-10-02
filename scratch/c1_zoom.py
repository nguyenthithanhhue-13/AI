"""Cắt phóng to vùng quanh robot của vài cảnh validation (mỗi kiểu vẽ, ưu tiên ảnh bị xuống cấp)."""
import sys
sys.path.insert(0, "src")
from PIL import Image
from common import *

scenes = load_split("validation")[2]
seen = {}
for i, s in enumerate(scenes):
    d = s["degradation"]
    if s["style"] not in seen and d["jpeg_quality"] and d["jpeg_quality"] < 55 and d["blur"] > 0:
        seen[s["style"]] = i
tiles = []
for st, i in seen.items():
    s = scenes[i]
    im = Image.open(DATA / "validation" / s["image"]).convert("RGB")
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    x, y = xy[tuple(s["robot"]["rc"])]
    x0 = int(min(max(x - 220, 0), s["width"] - 440)); y0 = int(min(max(y - 170, 0), s["height"] - 340))
    tiles.append(im.crop((x0, y0, x0 + 440, y0 + 340)).resize((880, 680), Image.LANCZOS))
    print(st, i, s["image"], s["degradation"], s["road_look"])
W = Image.new("RGB", (1760, 1360))
for k, t in enumerate(tiles):
    W.paste(t, ((k % 2) * 880, (k // 2) * 680))
W.save("scratch/zoom.png")
