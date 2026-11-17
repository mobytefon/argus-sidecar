"""Hot model reload: swap the classifier when the bundle files change.

`ModelStore.current` is the active `ArgusClassifier`. Request handlers read it
once per request, so a swap never affects a request already running; the old
model is freed once its last request finishes.

A watcher thread polls a signature of the bundle files (mtime_ns and size).
A new bundle is loaded only when the signature differs from the running
model's AND stayed the same across two consecutive polls, so a copy still in
progress is not picked up. A bundle that fails to load (bad ONNX contract,
bad threshold, broken Python, score out of range...) is logged and remembered;
the running model keeps serving and the same files are not retried until they
change again. SIGHUP (`request_reload`) triggers an immediate attempt.
"""

from __future__ import annotations

import logging
import os
import threading

from argus_sidecar.classifier import MODEL_FILES, ArgusClassifier

log = logging.getLogger(__name__)


def bundle_signature(model_dir: str) -> tuple:
    """(name, mtime_ns, size) per bundle file; (name, None, None) when missing."""
    sig = []
    for name in MODEL_FILES:
        try:
            st = os.stat(os.path.join(model_dir, name))
            sig.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((name, None, None))
    return tuple(sig)


class ModelStore:
    def __init__(
        self,
        model_dir: str,
        threshold="fpr0.1%",
        intra_threads: int = 1,
        interval_s: float = 5.0,
        factory=ArgusClassifier,
    ) -> None:
        self.model_dir = os.path.abspath(model_dir)
        self.threshold = threshold
        self.intra_threads = intra_threads
        self.interval_s = interval_s
        self._factory = factory

        self._lock = threading.Lock()  # one load attempt at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._reload_requested = False
        self._thread: threading.Thread | None = None

        # A broken bundle at startup raises: the process must not serve without a model.
        sig = bundle_signature(self.model_dir)
        self.current = self._build()
        self._loaded_sig = sig  # if files moved during this load, the next polls reload
        self._failed_sig = None
        self._pending_sig = None

    def _build(self):
        return self._factory(self.model_dir, threshold=self.threshold, intra_threads=self.intra_threads)

    def poll(self) -> bool:
        """One watcher step. Returns True when a new model was swapped in."""
        with self._lock:
            sig = bundle_signature(self.model_dir)
            if sig == self._loaded_sig or sig == self._failed_sig:
                self._pending_sig = None
                return False
            if sig != self._pending_sig:
                self._pending_sig = sig  # changed: wait for one more identical poll
                return False
            self._pending_sig = None
            return self._load(sig)

    def reload_now(self) -> bool:
        """Load the bundle now, without waiting for stable polls (SIGHUP)."""
        with self._lock:
            self._pending_sig = None
            return self._load(bundle_signature(self.model_dir))

    def _load(self, sig: tuple) -> bool:
        try:
            clf = self._build()
        except Exception as e:  # noqa: BLE001 — any failure keeps the running model
            self._failed_sig = sig
            log.error(
                "model reload failed, keeping the running model (version=%s run=%s): %s: %s",
                getattr(self.current, "version", None), getattr(self.current, "run", None),
                type(e).__name__, e,
            )
            return False
        if bundle_signature(self.model_dir) != sig:
            log.warning("model files changed while loading; discarding and waiting for them to settle")
            return False
        self.current = clf
        self._loaded_sig = sig
        self._failed_sig = None
        log.info("model reloaded version=%s run=%s threshold=%.6f",
                 getattr(clf, "version", None), getattr(clf, "run", None), clf.threshold)
        return True

    def request_reload(self) -> None:
        """Ask the watcher thread for an immediate reload (signal-handler safe)."""
        self._reload_requested = True
        self._wake.set()

    def start(self) -> None:
        """Start the watcher thread (also needed for SIGHUP when polling is off)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="argus-model-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        timeout = self.interval_s if self.interval_s > 0 else None
        while not self._stop.is_set():
            self._wake.wait(timeout)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                if self._reload_requested:
                    self._reload_requested = False
                    log.info("SIGHUP received, reloading the model")
                    self.reload_now()
                elif timeout is not None:
                    self.poll()
            except Exception:  # noqa: BLE001 — the watcher must survive anything
                log.exception("model watcher error")
