# syntax=docker/dockerfile:1.6
# Standalone build: docker build -t argus-sidecar:v0.2.2 .
# The model is mounted read-only at /model at runtime; it is not in the image
# (the repository's model/ directory holds the released bundle).
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS protogen
WORKDIR /build
RUN pip install --no-cache-dir grpcio-tools==1.84.0
COPY proto/ proto/
RUN mkdir -p generated && python -m grpc_tools.protoc -I proto --python_out=generated --grpc_python_out=generated proto/argus.proto && sed -i 's/^import argus_pb2/from . import argus_pb2/' generated/argus_pb2_grpc.py

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ARG PUID=10000
ARG PGID=10000
# /run/argus is world-writable with the sticky bit (like /tmp): a fresh socket
# volume takes this mode from whichever container mounts it first, and the
# sidecar may run as the image user or as 0:0 (rootless Docker, where it is
# the host user) with every capability dropped, so it can rely on no owner.
RUN groupadd -g ${PGID} argus && useradd -r -u ${PUID} -g argus -m argus && mkdir -p /run/argus /model && chmod 1777 /run/argus
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY argus_sidecar/ argus_sidecar/
COPY --from=protogen /build/generated/ argus_sidecar/_proto/
LABEL org.opencontainers.image.source="https://github.com/mobytefon/argus-sidecar" org.opencontainers.image.description="Argus Sentinel v4 ML sidecar for the Coraza mlcheck action" org.opencontainers.image.licenses="Apache-2.0"
USER argus
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ARGUS_MODEL_DIR=/model ARGUS_SOCKET=/run/argus/argus.sock ARGUS_SOCKET_MODE=0o660 ARGUS_THRESHOLD=fpr0.1% ARGUS_WORKERS=4 ARGUS_MAX_CONCURRENT_RPCS=16 ARGUS_RELOAD_INTERVAL_S=5 ARGUS_MIN_REMAINING_MS=5 ORT_INTRA_THREADS=1 LOG_LEVEL=INFO ARGUS_LOG_REQUESTS=all
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s --retries=3 CMD python -m argus_sidecar.healthcheck
CMD ["python", "-m", "argus_sidecar"]
