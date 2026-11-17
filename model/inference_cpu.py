"""
inference_cpu.py — Argus Sentinel v4 inference trên CPU (ONNX Runtime). Không cần torch.

Cài đặt:  pip install onnxruntime tokenizers numpy
Cần các file cùng thư mục:  model.onnx, tokenizer.json, config.json, preprocess.py

Dùng:
    from inference_cpu import ArgusSentinelV4
    clf = ArgusSentinelV4("path/to/argus_sentinel_v4", threshold="fpr0.5%")
    clf.predict_one("GET", "/item?id=1' or '1'='1", {"User-Agent": "Mozilla/5.0"})
    # -> {'attack_prob': 0.99..., 'is_attack': True}
    clf.predict_batch([{"method": "GET", "uri": "/", "headers": {...}, "body": ""}, ...])
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from preprocess import build_input  # noqa: E402


class ArgusSentinelV4:
    def __init__(self, model_dir: str = HERE, threshold: str | float = "fpr0.5%",
                 intra_threads: int = 1, model_file: str = "model.onnx") -> None:
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.thresholds = cfg["thresholds"]
        self.threshold = float(self.thresholds[threshold] if isinstance(threshold, str) else threshold)

        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(cfg["max_seq_len"])

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads          # 1 thread / session, scale bằng nhiều worker
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # không đốt CPU khi rảnh
        self.sess = ort.InferenceSession(os.path.join(model_dir, model_file), so,
                                         providers=["CPUExecutionProvider"])

    @staticmethod
    def to_text(method: str, uri: str, headers=None, body: str | bytes = "") -> str:
        """headers: dict hoặc list[(name, value)]; tên header không phân biệt hoa/thường."""
        h: dict[str, str] = {}
        for k, v in (headers.items() if isinstance(headers, dict) else (headers or [])):
            k = k.lower()
            h[k] = f"{h[k]}; {v}" if k in h else v
        if isinstance(body, (bytes, bytearray)):
            body = bytes(body[:4096]).decode("utf-8", "replace")
        return build_input(method, uri, h.get("user-agent", ""), h.get("referer", ""),
                           h.get("cookie", ""), body or "")

    def predict_texts(self, texts: list[str]) -> np.ndarray:
        # 1 request: encode() nhanh hơn encode_batch() (tránh overhead thread-pool của tokenizers)
        encs = [self.tok.encode(texts[0])] if len(texts) == 1 else self.tok.encode_batch(texts)
        ids = np.zeros((len(encs), max(len(e.ids) for e in encs)), np.int64)   # pad theo request dài nhất
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
        out = self.sess.run(None, {"input_ids": ids, "attention_mask": (ids != 0).astype(np.int64)})[0]
        return out[:, 0]

    def predict_one(self, method: str, uri: str, headers=None, body: str | bytes = "") -> dict:
        p = float(self.predict_texts([self.to_text(method, uri, headers, body)])[0])
        return {"attack_prob": p, "is_attack": p > self.threshold}

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        texts = [self.to_text(r["method"], r["uri"], r.get("headers"), r.get("body", "")) for r in requests]
        return [{"attack_prob": float(p), "is_attack": bool(p > self.threshold)} for p in self.predict_texts(texts)]


if __name__ == "__main__":
    clf = ArgusSentinelV4()
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
    for m, u, b in [("GET", "/products?id=42&sort=price", ""),
                    ("GET", "/products?id=42' UNION SELECT username,password FROM users--", ""),
                    ("POST", "/comment", "text=<script>alert(document.cookie)</script>"),
                    ("GET", "/download?file=../../../../etc/passwd", "")]:
        print(clf.predict_one(m, u, ua, b), m, u, b)
