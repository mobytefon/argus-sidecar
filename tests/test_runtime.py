"""Runtime checks that do not need private model artifacts.

Run inside the image (as CI does):
    docker run --rm --entrypoint python -e PYTHONPATH=/app -v "$PWD/tests:/tests:ro" \
        argus-sidecar:test -m unittest discover -s /tests -p test_runtime.py -v
"""
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

import grpc
from argus_sidecar.classifier import ArgusClassifier, resolve_threshold
from argus_sidecar.model_store import ModelStore
from argus_sidecar.server import ArgusServicer
from argus_sidecar.socket_lock import socket_lease
from argus_sidecar._proto import argus_pb2

# A stand-in for the released inference_cpu.py: same import shape
# (`sys.path.insert(0, HERE); from preprocess import ...`) and same API, with a
# fake ONNX session. The probability comes from config.json; model.onnx must
# contain exactly b"fake-onnx".
FAKE_INFERENCE = '''
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from preprocess import build_input, MARK


class _Spec:
    def __init__(self, name, type_, shape):
        self.name, self.type, self.shape = name, type_, shape


class _Session:
    def get_inputs(self):
        return [_Spec("input_ids", "tensor(int64)", ["batch", "seq"]),
                _Spec("attention_mask", "tensor(int64)", ["batch", "seq"])]

    def get_outputs(self):
        return [_Spec("attack_prob", "tensor(float)", ["batch", 1])]


class ArgusSentinelV4:
    def __init__(self, model_dir, threshold="fpr0.5%", intra_threads=1, model_file="model.onnx"):
        with open(os.path.join(model_dir, model_file), "rb") as f:
            if f.read() != b"fake-onnx":
                raise RuntimeError("corrupt model.onnx")
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.prob = float(cfg["fake_prob"])
        self.threshold = float(threshold)
        self.sess = _Session()
        self.mark = MARK

    def predict_one(self, method, uri, headers=None, body=b""):
        build_input(method, uri)
        return {"attack_prob": self.prob, "is_attack": self.prob > self.threshold}
'''


def fake_preprocess(mark):
    return f"MARK = {mark!r}\n\ndef build_input(method, uri, *rest):\n    return MARK + method + uri\n"


class FakeBundle:
    """A five-file bundle in a temp dir with deterministic, distinct mtimes."""

    def __init__(self, root, prob=0.2, thresholds=None, mark="v1", run="run-1"):
        self.dir = Path(root)
        self._mtime = 1_700_000_000_000_000_000
        self.write("model.onnx", b"fake-onnx")
        self.write("tokenizer.json", "{}")
        self.write("preprocess.py", fake_preprocess(mark))
        self.write("inference_cpu.py", FAKE_INFERENCE)
        self.config(prob=prob, thresholds=thresholds, run=run)

    def write(self, name, content):
        path = self.dir / name
        if isinstance(content, str):
            content = content.encode("utf-8")
        path.write_bytes(content)
        # Filesystems may not tick between fast writes: force a new mtime.
        self._mtime += 1_000_000_000
        os.utime(path, ns=(self._mtime, self._mtime))

    def config(self, prob=0.2, thresholds=None, run="run-1"):
        cfg = {"version": "v4-binary", "run": run, "max_seq_len": 256,
               "thresholds": thresholds or {"fpr0.1%": 0.9, "0.5": 0.5},
               "fake_prob": prob}
        self.write("config.json", json.dumps(cfg))


class RuntimeTests(unittest.TestCase):
    def test_threshold_validation(self):
        self.assertEqual(resolve_threshold({"strict": 0.99}, "strict"), 0.99)
        self.assertEqual(resolve_threshold({}, "0.5"), 0.5)
        for invalid in ("bad", "", math.nan, math.inf, -0.1, 1.1):
            with self.subTest(value=invalid), self.assertRaises((RuntimeError, ValueError)):
                resolve_threshold({}, invalid)

    def test_socket_lock_excludes_second_server_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "argus.sock"
            with socket_lease(str(path)):
                with self.assertRaises(RuntimeError):
                    with socket_lease(str(path)):
                        self.fail("two servers acquired the same socket")
            path.write_text("do not delete")
            with self.assertRaises(RuntimeError):
                with socket_lease(str(path)):
                    pass
            self.assertEqual(path.read_text(), "do not delete")

    def test_stale_socket_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "argus.sock")
            with socket.socket(socket.AF_UNIX) as stale:
                stale.bind(path)
            with socket_lease(path):
                self.assertFalse(os.path.exists(path))

    def test_expired_deadline_skips_inference(self):
        classifier, context = Mock(), Mock()
        context.is_active.return_value = True
        context.time_remaining.return_value = 0.001
        context.abort.side_effect = RuntimeError("abort")
        with self.assertRaises(RuntimeError):
            ArgusServicer(Mock(current=classifier), 0.005).Classify(argus_pb2.ClassifyRequest(), context)
        context.abort.assert_called_once()
        self.assertEqual(context.abort.call_args.args[0], grpc.StatusCode.DEADLINE_EXCEEDED)
        classifier.classify.assert_not_called()

    def test_binary_wire_contract(self):
        classifier, context = Mock(), Mock()
        context.time_remaining.return_value = 1
        classifier.classify.return_value = {
            "is_clean": False, "risk_score": 0.999, "detections": {"attack": 0.999}}
        response = ArgusServicer(Mock(current=classifier), 0.005).Classify(
            argus_pb2.ClassifyRequest(method="POST", uri="/", body=b"raw"), context)
        self.assertFalse(response.is_clean)
        self.assertEqual(response.risk_score, 0.999)
        self.assertEqual(dict(response.detections), {"attack": 0.999})
        self.assertEqual(classifier.classify.call_args.kwargs["body"], b"raw")

    def test_servicer_uses_the_current_model_per_request(self):
        first, second, context = Mock(), Mock(), Mock()
        context.time_remaining.return_value = 1
        for clf, p in ((first, 0.1), (second, 0.2)):
            clf.classify.return_value = {"is_clean": True, "risk_score": p, "detections": {}}
        store = Mock(current=first)
        servicer = ArgusServicer(store, 0.005)
        self.assertAlmostEqual(servicer.Classify(argus_pb2.ClassifyRequest(), context).risk_score, 0.1)
        store.current = second
        self.assertAlmostEqual(servicer.Classify(argus_pb2.ClassifyRequest(), context).risk_score, 0.2)


class ModelLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="argus-bundle-")
        self.addCleanup(self._tmp.cleanup)
        self.bundle = FakeBundle(self._tmp.name)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def store(self, threshold="fpr0.1%"):
        calls = []

        def factory(*args, **kw):
            calls.append(1)
            return ArgusClassifier(*args, **kw)

        store = ModelStore(self.bundle.dir, threshold=threshold, interval_s=0, factory=factory)
        store.calls = calls
        return store

    def score(self, store):
        return store.current.classify("GET", "/", {}, b"")

    def test_first_load(self):
        store = self.store()
        self.assertEqual(store.current.threshold, 0.9)
        self.assertEqual(store.current.version, "v4-binary")
        self.assertEqual(store.current.run, "run-1")
        verdict = self.score(store)
        self.assertTrue(verdict["is_clean"])
        self.assertAlmostEqual(verdict["risk_score"], 0.2)
        self.assertNotIn("preprocess", sys.modules)
        self.assertFalse(store.poll(), "unchanged files must not reload")
        self.assertEqual(len(store.calls), 1)

    def test_broken_bundle_at_startup_raises(self):
        self.bundle.write("model.onnx", b"corrupt")
        with self.assertRaises(RuntimeError):
            self.store()

    def test_change_is_picked_up_only_after_two_stable_polls(self):
        store = self.store()
        old = store.current
        self.bundle.config(prob=0.3, run="run-2")
        self.assertFalse(store.poll(), "first poll only notices the change")
        self.assertIs(store.current, old)
        self.bundle.config(prob=0.4, run="run-3")  # still being copied
        self.assertFalse(store.poll())
        self.assertIs(store.current, old)
        self.assertTrue(store.poll(), "second identical poll loads")
        self.assertEqual(store.current.run, "run-3")
        self.assertAlmostEqual(self.score(store)["risk_score"], 0.4)
        self.assertFalse(store.poll())

    def test_broken_file_keeps_the_old_model_serving(self):
        store = self.store()
        old = store.current
        self.bundle.write("model.onnx", b"corrupt")
        self.assertFalse(store.poll())
        self.assertFalse(store.poll())
        self.assertIs(store.current, old)
        self.assertAlmostEqual(self.score(store)["risk_score"], 0.2)
        attempts = len(store.calls)
        for _ in range(3):
            self.assertFalse(store.poll())
        self.assertEqual(len(store.calls), attempts, "a failed bundle is not retried until it changes")

        self.bundle.write("model.onnx", b"fake-onnx")
        store.poll()
        self.assertTrue(store.poll())
        self.assertIsNot(store.current, old)

    def test_out_of_range_score_is_rejected_on_reload(self):
        store = self.store()
        old = store.current
        self.bundle.config(prob=1.5)
        store.poll()
        self.assertFalse(store.poll())
        self.assertIs(store.current, old)

    def test_broken_python_keeps_the_old_model_serving(self):
        store = self.store()
        old = store.current
        self.bundle.write("preprocess.py", "def build_input(:\n")
        store.poll()
        self.assertFalse(store.poll())
        self.assertIs(store.current, old)
        self.assertNotIn("preprocess", sys.modules)

    def test_preset_threshold_comes_from_the_new_config(self):
        store = self.store()
        self.bundle.config(prob=0.7)
        store.poll()
        self.assertTrue(store.poll())
        self.assertTrue(self.score(store)["is_clean"], "0.7 is below the old fpr0.1% preset 0.9")
        self.bundle.config(prob=0.7, thresholds={"fpr0.1%": 0.6})
        store.poll()
        self.assertTrue(store.poll())
        self.assertEqual(store.current.threshold, 0.6)
        verdict = self.score(store)
        self.assertFalse(verdict["is_clean"])
        self.assertEqual(set(verdict["detections"]), {"attack"})

    def test_two_loads_do_not_share_a_preprocess_module(self):
        sentinel = object()
        sys.modules["preprocess"] = sentinel  # someone else's module named preprocess
        self.addCleanup(sys.modules.pop, "preprocess", None)
        store = self.store()
        first = store.current
        self.bundle.write("preprocess.py", fake_preprocess("v2"))
        store.poll()
        self.assertTrue(store.poll())
        second = store.current
        self.assertEqual(first.model.mark, "v1")
        self.assertEqual(second.model.mark, "v2")
        self.assertIsNot(first._v4.build_input, second._v4.build_input)
        self.assertIs(sys.modules["preprocess"], sentinel, "sys.modules['preprocess'] is restored")

    def test_files_changed_during_load_are_discarded(self):
        bundle = self.bundle

        def factory(*args, **kw):
            clf = ArgusClassifier(*args, **kw)
            if factory.mutate:
                factory.mutate = False
                bundle.config(prob=0.6, run="run-late")
            return clf
        factory.mutate = False

        store = ModelStore(bundle.dir, interval_s=0, factory=factory)
        old = store.current
        bundle.config(prob=0.5, run="run-2")
        store.poll()
        factory.mutate = True
        self.assertFalse(store.poll(), "files moved while loading")
        self.assertIs(store.current, old)
        store.poll()
        self.assertTrue(store.poll())
        self.assertEqual(store.current.run, "run-late")

    def test_reload_now_skips_the_stability_wait(self):
        store = self.store()
        self.bundle.config(prob=0.3, run="run-hup")
        self.assertTrue(store.reload_now())
        self.assertEqual(store.current.run, "run-hup")

    def wait_for_run(self, store, run):
        deadline = time.monotonic() + 5
        while store.current.run != run and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(store.current.run, run)

    def test_watcher_thread_polls(self):
        store = ModelStore(self.bundle.dir, interval_s=0.02)
        store.start()
        self.addCleanup(store.stop)
        self.bundle.config(prob=0.3, run="run-polled")
        self.wait_for_run(store, "run-polled")

    def test_reload_request_works_with_polling_disabled(self):
        store = ModelStore(self.bundle.dir, interval_s=0)  # polling off: SIGHUP only
        store.start()
        self.addCleanup(store.stop)
        self.bundle.config(prob=0.3, run="run-hup")
        time.sleep(0.05)
        self.assertEqual(store.current.run, "run-1")
        store.request_reload()
        self.wait_for_run(store, "run-hup")


if __name__ == "__main__":
    unittest.main()
