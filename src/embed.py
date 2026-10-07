"""Vector nghĩa của câu/cụm tiếng Việt bằng mô hình pretrained công khai multilingual-e5-small (118 triệu tham số).

Chạy OFFLINE bằng onnxruntime sau khi đã tải mô hình một lần về outputs/e5_small/:
    python src/embed.py download
Không gọi API nào khi dự đoán.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from common import CACHE, OUT

MODEL_DIR = OUT / "e5_small"
REPO = "intfloat/multilingual-e5-small"


def download():
    from huggingface_hub import hf_hub_download
    MODEL_DIR.mkdir(exist_ok=True)
    for f in ("onnx/model.onnx", "tokenizer.json"):
        p = hf_hub_download(REPO, f, local_dir=MODEL_DIR)
        print("đã tải", p)


class Embedder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")
        self.tok.enable_truncation(max_length=64)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(str(MODEL_DIR / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.cache_path = CACHE / "e5_cache.pkl"
        self.cache = pickle.load(open(self.cache_path, "rb")) if self.cache_path.exists() else {}
        self._dirty = 0

    def encode(self, texts, batch=64, prefix="query: "):
        """-> mảng (n, 384) đã chuẩn hóa độ dài. Có cache theo chuỗi (gồm cả tiền tố "query: " / "passage: " của e5)."""
        texts = [prefix + t for t in texts]
        todo = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc = self.tok.encode_batch(chunk)
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / m.sum(1)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            for t, x in zip(chunk, v):
                self.cache[t] = x.astype(np.float32)
            self._dirty += len(chunk)
        return np.array([self.cache[t] for t in texts])

    def save(self):
        """Ghi nguyên khối (file tạm + thay thế): nhiều tiến trình cùng ghi cache không làm hỏng file."""
        if self._dirty:
            import os
            tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f)
            try:
                os.replace(tmp, self.cache_path)
            except OSError:                    # file đang bị tiến trình khác mở: bỏ qua, cache chỉ để chạy nhanh hơn
                os.remove(tmp)
            self._dirty = 0


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    e = Embedder()
    v = e.encode(["thư viện", "nơi mượn sách", "bãi đỗ xe", "gấp lắm", "khẩn trương lên", "không vội đâu"])
    print(np.round(v @ v.T, 2))
    e.save()
