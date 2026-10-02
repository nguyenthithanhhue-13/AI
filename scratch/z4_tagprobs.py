import sys
sys.path.insert(0, "src")
from common import *
from nlp2 import *

trs = load_split("train")[2]
p = MissionParser2().fit([s["mission"] for s in trs])
import pickle
pickle.dump(p, open("cache/nlp2_trainonly.pkl", "wb"))
for sent in ["hang cho noi kham suc khoe man trai : buu kien", "dich den la san luyen tap , hang can giao la bo vot",
             "ghe phong an tap the lay nguyen lieu nau an truoc , roi giao hoa chat den phong thi nghiem phia nam giup minh",
             "tui do phai duoc mang toi khu van dong", "nho ghe phong thuc nghiem nam xa phong tai lieu lay hoa chat",
             "nhung phai tat qua san bong nam gan phong giao vu truoc da", "co don giao hop sach tra o noi muon sach tham khao cach xa tram xa",
             "chua di thang duoc : ghe noi nop ho so nam gan ham xe truoc , roi dua ket nuoc ngot den nha an giup minh"]:
    tt = tag_tokens(sent)
    pr = p.tagger.predict(tt)
    print(" ".join(f"{t}[{q:.2f}]" for t, q in zip(tt, pr)))
