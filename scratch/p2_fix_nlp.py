"""Sửa src/nlp.py (vòng private): ranh giới câu + khung câu gây nhiễu / đính chính mới. Viết bằng file để tránh lỗi thoát ký tự."""
p = "src/nlp.py"
s = open(p, encoding="utf-8").read()
lines = s.split("\n")
for i, l in enumerate(lines):
    if "\x01" in l:
        lines[i] = '    t = re.sub(r",\\s*(doi lai|thay vao do|dung ra)\\s*:", r". \\1:", t)'
s = "\n".join(lines)
old_prefix = 'PREFIX = r"^(robot oi|yeu cau moi|nho ban nhe|xin chao|chao robot|nhan robot)\\b[,: ]*"'
new_prefix = 'PREFIX = r"^(robot oi|yeu cau moi|nho ban nhe|xin chao|chao robot|nhan robot|doi lai|thay vao do|dung ra)\\b[,: ]*"'
assert old_prefix in s
s = s.replace(old_prefix, new_prefix)
old_dis = 'r"^nguoi nhan da roi (.+) roi$", r"^(.+) khong phai diem nhan$", r"^bo qua (.+), khong phai o do$"]'
new_dis = ('r"^nguoi nhan da roi (.+) roi$", r"^(.+) khong phai diem nhan$", r"^bo qua (.+), khong phai o do$",\n'
           '              # (vòng private) đính chính / hủy đơn: "(tin trước ghi X là nhầm)", "Lúc nãy nhắn nhầm là X. Đúng ra: ...",\n'
           '              # "Hủy đơn giao X. Thay vào đó: ...", "Không giao X nữa nhé, đổi lại: ..."\n'
           '              r"^tin truoc ghi (.+) la nham$", r"^luc nay nhan nham la (.+)$", r"^huy don giao (.+)$",\n'
           '              r"^khong giao (.+) nua nhe$"]')
assert old_dis in s
s = s.replace(old_dis, new_dis)
assert "\x01" not in s and "\x08" not in s
open(p, "w", encoding="utf-8").write(s)
print("ok")
