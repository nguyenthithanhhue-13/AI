"""Thay một mục trong GIAI_THICH.md bằng nội dung mới (từ tiêu đề start tới ngay trước tiêu đề end)."""
import sys
from pathlib import Path

doc = Path("GIAI_THICH.md")
start, end, src = sys.argv[1], sys.argv[2], Path(sys.argv[3])
t = doc.read_text(encoding="utf-8")
a = t.index(start)
b = t.index(end) if end != "EOF" else len(t)
doc.write_text(t[:a] + src.read_text(encoding="utf-8") + t[b:], encoding="utf-8")
print("replaced", b - a, "chars with", len(src.read_text(encoding="utf-8")))
