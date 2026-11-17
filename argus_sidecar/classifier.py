"""Argus Sentinel v4 classifier wrapper for the sidecar.

The v4 model is a binary clean/attack classifier. The sidecar keeps the
pre-v4 gRPC contract (see proto/argus.proto, the single source of truth shared
with the Go action) and maps v4 outputs onto it:

    attack_prob -> risk_score
    attack_prob > threshold -> is_clean = False, detections = {"attack": p}

The v4 preprocessing and inference modules (`preprocess.py`,
`inference_cpu.py`) ship with the model bundle and are loaded from the model
directory, so the normalization contract is never re-implemented here. Each
load imports them afresh under unique module names, so a hot reload never
reuses code from the previous bundle.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import math
import os
import sys
import threading

log = logging.getLogger(__name__)

# The minimal v4 bundle. Anything else (datasets, training runs, .env files)
# must never be part of the image. The hot-reload watcher tracks these files.
MODEL_FILES = (
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "preprocess.py",
    "inference_cpu.py",
)

# Bundle imports touch process-wide state (sys.modules["preprocess"], sys.path),
# so they are serialized.
_IMPORT_LOCK = threading.Lock()
_LOAD_SEQ = itertools.count(1)
_MISSING = object()


def resolve_threshold(thresholds: dict, spec) -> float:
    """Resolve ARGUS_THRESHOLD: a preset name ("fpr0.1%") or a float in [0,1]."""
    if isinstance(spec, str):
        name = spec.strip()
        if name in thresholds:
            value = float(thresholds[name])
        else:
            try:
                value = float(name)
            except ValueError:
                raise RuntimeError(
                    f"ARGUS_THRESHOLD={spec!r} is neither a preset "
                    f"({', '.join(sorted(map(str, thresholds)))}) nor a float in [0,1]"
                ) from None
    else:
        value = float(spec)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"threshold {value!r} is not a finite float in [0,1]")
    return value


def _exec_module(name: str, path: str):
    """Import `path` as module `name`, always from source.

    Compiling the source directly (instead of spec.loader.exec_module) skips
    __pycache__, whose validity check (mtime + size) could otherwise hand back
    bytecode of a previous bundle.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    with open(path, "rb") as f:
        code = compile(f.read(), path, "exec", dont_inherit=True)
    sys.modules[name] = module
    exec(code, module.__dict__)  # noqa: S102 — operator-supplied bundle code, by design
    return module


def _load_v4_modules(model_dir: str):
    """Import the bundle's preprocess.py and inference_cpu.py in isolation.

    inference_cpu.py does `sys.path.insert(0, HERE); from preprocess import
    build_input`. While it executes, sys.modules["preprocess"] points at this
    load's own preprocess module, so the import resolves to it and never to a
    module cached by an earlier load. Process-wide state is restored afterwards.
    """
    seq = next(_LOAD_SEQ)
    pre_name = f"_argus_bundle{seq}_preprocess"
    inf_name = f"_argus_bundle{seq}_inference_cpu"
    with _IMPORT_LOCK:
        saved_path = list(sys.path)
        saved_preprocess = sys.modules.get("preprocess", _MISSING)
        try:
            preprocess = _exec_module(pre_name, os.path.join(model_dir, "preprocess.py"))
            sys.modules["preprocess"] = preprocess
            return _exec_module(inf_name, os.path.join(model_dir, "inference_cpu.py"))
        finally:
            if saved_preprocess is _MISSING:
                sys.modules.pop("preprocess", None)
            else:
                sys.modules["preprocess"] = saved_preprocess
            # The modules stay alive through the objects that use them; dropping
            # the names keeps repeated reloads from accumulating in sys.modules.
            sys.modules.pop(pre_name, None)
            sys.modules.pop(inf_name, None)
            sys.path[:] = saved_path


def _check_prob(p) -> float:
    p = float(p)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"model returned an invalid attack_prob: {p!r}")
    return p


class ArgusClassifier:
    """Thread-safe v4 classifier: load once, classify() per request."""

    def __init__(self, model_dir: str, threshold="fpr0.1%", intra_threads: int = 1) -> None:
        model_dir = os.path.abspath(model_dir)
        self.model_dir = model_dir
        self.threshold_spec = threshold

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        if not isinstance(self.config, dict):
            raise RuntimeError("config.json must be a JSON object")
        self.version = self.config.get("version")
        self.run = self.config.get("run")

        model_file = (self.config.get("onnx") or {}).get("file", "model.onnx")
        if model_file != "model.onnx":
            raise RuntimeError(f"config.json points at an unexpected model file {model_file!r}")

        thresholds = self.config.get("thresholds") or {}
        self.threshold = resolve_threshold(thresholds, threshold)

        if not 1 <= intra_threads <= 256:
            raise ValueError("ORT_INTRA_THREADS must be 1..256")
        self._v4 = _load_v4_modules(model_dir)
        self.model = self._v4.ArgusSentinelV4(
            model_dir,
            threshold=self.threshold,
            intra_threads=intra_threads,
            model_file=model_file,
        )
        self._validate_session()
        self.warmup()

        log.info(
            "argus model ready: version=%s run=%s threshold=%r (spec=%s) max_seq_len=%s",
            self.version, self.run, self.threshold, threshold, self.config.get("max_seq_len"),
        )

    def _validate_session(self) -> None:
        """Fail fast when the ONNX graph does not match the v4 contract."""
        sess = self.model.sess
        inputs = {i.name: i for i in sess.get_inputs()}
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise RuntimeError(f"unexpected ONNX inputs {sorted(inputs)}, expected input_ids/attention_mask")
        for name in ("input_ids", "attention_mask"):
            if inputs[name].type != "tensor(int64)":
                raise RuntimeError(f"model.onnx input {name!r} is {inputs[name].type}, expected tensor(int64)")
        output_specs = sess.get_outputs()
        outputs = [o.name for o in output_specs]
        if (outputs != ["attack_prob"] or output_specs[0].type != "tensor(float)"
                or len(output_specs[0].shape) != 2 or output_specs[0].shape[1] != 1):
            raise RuntimeError("expected ONNX output attack_prob float [batch,1]")

    def warmup(self) -> None:
        """Run the graph so the first real request is not slow, and check its scores."""
        for method, body in (("GET", b""), ("POST", b"warmup")):
            result = self.model.predict_one(method, "/healthz", {"User-Agent": "argus-warmup/1.0"}, body)
            _check_prob(result["attack_prob"])

    def classify(self, method: str, uri: str, headers, body: bytes) -> dict:
        """Classify one request; returns the pre-v4 response contract.

        headers: mapping of lower-cased name -> value (duplicates already
        joined with "; " by the Go action). body: raw bytes (the action caps
        at 4 KiB; the v4 preprocessing further caps each field).
        """
        result = self.model.predict_one(method, uri, dict(headers), bytes(body))
        p = _check_prob(result["attack_prob"])
        is_attack = bool(result["is_attack"])
        return {
            "is_clean": not is_attack,
            "risk_score": p,
            "detections": {"attack": p} if is_attack else {},
        }
