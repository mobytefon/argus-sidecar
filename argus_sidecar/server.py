"""gRPC server for the Argus Sentinel v4 sidecar.

Binds an AF_UNIX socket so the Coraza `mlcheck` action can call us with
near-zero overhead (no TCP, no TLS, no network stack). The server keeps the
pre-v4 operational shape (synchronous handlers on a thread pool) and adds the
v4 availability guards:

  * gRPC-level concurrency cap: requests above `ARGUS_MAX_CONCURRENT_RPCS`
    are rejected immediately with RESOURCE_EXHAUSTED instead of queueing
    without bound (the action counts these as `overload` and fails open; keep
    the action's ARGUS_MAX_INFLIGHT at or below this cap);
  * a deadline check before inference: a request whose client budget is
    nearly gone is dropped with DEADLINE_EXCEEDED instead of competing for
    CPU with work that can still make it.

The model is hot-reloaded from the model directory (see model_store.py).
"""

from __future__ import annotations

import logging
import math
import os
import signal
from concurrent import futures

import grpc

from argus_sidecar._proto import argus_pb2, argus_pb2_grpc
from argus_sidecar.model_store import ModelStore
from argus_sidecar.socket_lock import socket_lease

log = logging.getLogger("argus_sidecar.server")


def unix_target(socket_path: str) -> str:
    """gRPC client target for a UDS path (POSIX and Windows-drive friendly)."""
    return "unix://" + socket_path if socket_path.startswith("/") else "unix:" + socket_path


class ArgusServicer(argus_pb2_grpc.ArgusSentinelServicer):
    """store: any object whose `.current` is the active classifier."""

    def __init__(self, store, min_remaining_s: float) -> None:
        self.store = store
        self.min_remaining_s = min_remaining_s

    def Classify(self, request, context):  # noqa: N802 (gRPC naming convention)
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "request cancelled")
        remaining = context.time_remaining()
        if remaining is not None and remaining < self.min_remaining_s:
            context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                f"deadline almost expired ({remaining:.4f}s left)",
            )
            return  # unreachable, abort() raises

        clf = self.store.current  # read once: a concurrent reload cannot split this request
        try:
            verdict = clf.classify(
                method=request.method,
                uri=request.uri,
                headers=dict(request.headers),
                body=bytes(request.body),
            )
        except Exception as e:  # noqa: BLE001 — surface every error to the client
            log.error("Classify failed: %s", type(e).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "inference failed")
            return  # unreachable, abort() raises

        return argus_pb2.ClassifyResponse(
            is_clean=verdict["is_clean"],
            risk_score=verdict["risk_score"],
            detections=verdict["detections"],
        )


def serve() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    model_dir = os.getenv("ARGUS_MODEL_DIR", "/model")
    socket_path = os.getenv("ARGUS_SOCKET", "/run/argus/argus.sock")
    socket_mode = int(os.getenv("ARGUS_SOCKET_MODE", "0o660"), 0)
    workers = int(os.getenv("ARGUS_WORKERS", "4"))
    max_concurrent = int(os.getenv("ARGUS_MAX_CONCURRENT_RPCS", "16"))
    min_remaining_ms = int(os.getenv("ARGUS_MIN_REMAINING_MS", "5"))
    threshold_spec = os.getenv("ARGUS_THRESHOLD", "fpr0.1%")
    reload_interval_s = float(os.getenv("ARGUS_RELOAD_INTERVAL_S", "5"))

    if not 1 <= workers <= 256 or not 1 <= max_concurrent <= 1024:
        raise ValueError("ARGUS_WORKERS must be 1..256 and ARGUS_MAX_CONCURRENT_RPCS 1..1024")
    if not 0 <= min_remaining_ms <= 60000:
        raise ValueError("ARGUS_MIN_REMAINING_MS must be 0..60000")
    if socket_mode not in (0o600, 0o660):
        raise ValueError("ARGUS_SOCKET_MODE must be 0o600 or 0o660")
    if not math.isfinite(reload_interval_s) or not 0 <= reload_interval_s <= 3600:
        raise ValueError("ARGUS_RELOAD_INTERVAL_S must be 0..3600 (0 disables polling)")

    # Exits the process when the bundle at startup is broken.
    store = ModelStore(
        model_dir,
        threshold=threshold_spec,
        intra_threads=int(os.getenv("ORT_INTRA_THREADS", "1")),
        interval_s=reload_interval_s,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        maximum_concurrent_rpcs=max_concurrent,
        options=[
            ("grpc.max_send_message_length", 1 << 20),       # 1 MiB
            ("grpc.max_receive_message_length", 8 << 20),    # 8 MiB
            ("grpc.so_reuseport", 0),
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )
    argus_pb2_grpc.add_ArgusSentinelServicer_to_server(
        ArgusServicer(store, min_remaining_ms / 1000.0), server
    )

    with socket_lease(socket_path):
        bind = "unix:" + socket_path
        if server.add_insecure_port(bind) == 0:
            raise RuntimeError("cannot bind Argus Unix socket")
        # Permission errors fail startup instead of exposing a misconfigured socket.
        os.chmod(socket_path, socket_mode)
        server.start()
        store.start()
        log.info("listening on %s workers=%d max_concurrent=%d threshold=%.6f reload_interval_s=%g",
                 bind, workers, max_concurrent, store.current.threshold, reload_interval_s)

        def shutdown(signum, _frame):
            log.info("signal %d received, draining", signum)
            store.stop()
            server.stop(grace=2.0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGHUP, lambda _signum, _frame: store.request_reload())
        try:
            server.wait_for_termination()
        finally:
            store.stop()
            server.stop(grace=2.0).wait()


if __name__ == "__main__":
    serve()
