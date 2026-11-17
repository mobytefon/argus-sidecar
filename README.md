# Argus Sentinel v4 sidecar

Python/ONNX CPU inference service for the Coraza `mlcheck` action in
[mobytefon/coraza-mlcheck](https://github.com/mobytefon/coraza-mlcheck).
The action calls it over a Unix socket (`/run/argus/argus.sock`) on a shared
Docker volume. The classifier is binary: `attack_prob > threshold`; the gRPC
response uses `risk_score`, `is_clean` and an optional `attack` label. It does
not label SQLi/XSS.

## Ubuntu 24.04 deployment

Requirements: Docker Engine with Compose v2, and your trusted Argus Sentinel
v4 model bundle. This repository and its public image contain **no model
weights, tokenizer or training data**. Containers are Linux amd64; other
architectures need their own build and validation.

The model directory holds exactly the five bundle files: `model.onnx`,
`tokenizer.json`, `config.json`, `preprocess.py` and `inference_cpu.py`.
The two Python files are executed by the sidecar: only mount bundles from a
source you trust.

```sh
git clone --branch v0.2.0 https://github.com/mobytefon/argus-sidecar.git
cd argus-sidecar
sudo install -d -m 0750 -o 10000 -g 10000 /opt/argus/models/v4
sudo install -m 0640 -o 10000 -g 10000 \
  /path/to/argus_sentinel_v4/{model.onnx,tokenizer.json,config.json,preprocess.py,inference_cpu.py} \
  /opt/argus/models/v4/
cp .env.example .env
# Edit .env to match your model directory and Ingressi socket settings.
docker compose config --quiet
docker compose pull        # or: docker compose build --pull
docker compose up -d
docker compose exec argus-sidecar python -m argus_sidecar.healthcheck
```

Keep `.env` in the sidecar checkout: Compose reads it again after a new SSH
login, without shell exports. It is gitignored. Both `ARGUS_SOCKET_VOLUME`
and `ARGUS_SOCKET` must match Ingressi's `.env`; custom socket paths must
remain under the shared `/run/argus` mount. The model path in `.env` is a
**host path**; the container always reads `/model` (mounted read-only).

The image is published as `ghcr.io/mobytefon/argus-sidecar:v0.2.0`. The
container runs with `network_mode: none`: it only talks over the socket.

Mount the **same named volume** at `/run/argus` in Caddy — the Ingressi ML
overlay references it as an external volume with the same name
(`ARGUS_SOCKET_VOLUME`), so the two stacks share the socket without sharing
a Compose project. Both images default to UID/GID 10000:10000. For custom
Caddy IDs, rebuild with matching `--build-arg PUID=... --build-arg PGID=...`.
Socket mode is 0660. For an existing volume, verify ownership once; for a
fresh volume start the sidecar first. No TCP port is exposed. One sidecar per
Caddy node; do not share one socket filesystem across hosts.

The Ingressi ML overlay does **not** run a sidecar: this stack (or an
equivalent service manager) is the only sidecar deployment, and the overlay
consumes its socket volume. Run exactly one sidecar per Caddy node — a lock
prevents two sidecars from unlinking each other's live socket. See the
[coraza-mlcheck README](https://github.com/mobytefon/coraza-mlcheck) for the
Caddy side (rules, detect mode, client settings).

## Hot model swap

The sidecar watches the five bundle files and swaps the model without a
restart and without dropping requests:

1. Copy the new files over the host model directory (`ARGUS_MODEL_DIR_HOST`),
   readable by UID 10000:

   ```sh
   sudo install -m 0640 -o 10000 -g 10000 \
     /path/to/new_bundle/{model.onnx,tokenizer.json,config.json,preprocess.py,inference_cpu.py} \
     /opt/argus/models/v4/
   ```

2. Watch the log: `docker compose logs -f argus-sidecar`. Within about two
   `ARGUS_RELOAD_INTERVAL_S` periods after the last file stops changing you
   see `model reloaded version=… run=… threshold=…`. To reload immediately
   instead of waiting: `docker compose kill -s HUP argus-sidecar`.

How it behaves:

- Every `ARGUS_RELOAD_INTERVAL_S` seconds (default 5; `0` disables polling)
  the sidecar compares the files' modification time and size with the running
  model's. A change is loaded only once the files are identical across two
  consecutive checks, so a copy in progress is not picked up half-written.
- The new bundle is fully loaded and checked before it serves: ONNX contract
  (`input_ids`/`attention_mask` int64 → `attack_prob` float `[batch,1]`),
  threshold, warmup inference with scores in [0,1]. A preset
  `ARGUS_THRESHOLD` such as `fpr0.1%` is re-resolved from the **new**
  `config.json`. `preprocess.py` and `inference_cpu.py` are imported afresh
  for each load, never reused from the old bundle.
- If the files change again while loading, the result is discarded and the
  sidecar waits for them to settle.
- If the new bundle fails, the old model keeps serving and the error is
  logged (`model reload failed, keeping the running model …`). The same
  files are not retried until they change again (or on SIGHUP). Fix the
  files and copy them again.
- Requests already running finish on the model they started with.
- A broken bundle at **startup** still exits the process (Compose restarts it).

Rollback: copy the previous bundle back the same way. Keep it around.

## Configuration

| Environment | Default | Meaning |
|---|---|---|
| ARGUS_MODEL_DIR_HOST | required | Host bundle path in .env; mounted read-only at /model |
| ARGUS_SOCKET_VOLUME | ingressi-argus-run | Shared Docker volume name; match Ingressi |
| ARGUS_MODEL_DIR | /model | Fixed container mount in Compose |
| ARGUS_THRESHOLD | fpr0.1% | Preset from model config, or finite float 0..1 |
| ARGUS_RELOAD_INTERVAL_S | 5 | Seconds between model-file checks, 0..3600; 0 disables polling (SIGHUP still reloads) |
| ARGUS_SOCKET | /run/argus/argus.sock | Absolute Unix socket path |
| ARGUS_SOCKET_MODE | 0o660 | 0o600 or 0o660 |
| ARGUS_WORKERS | 4 | Inference worker threads, 1..256 |
| ARGUS_MAX_CONCURRENT_RPCS | 16 | Bounded active/queued RPC count, 1..1024 |
| ARGUS_MIN_REMAINING_MS | 5 | Reject requests with almost-expired deadlines |
| ORT_INTRA_THREADS | 1 | ONNX threads per session |
| ARGUS_LOG_LEVEL | INFO | Forwarded to the runtime as LOG_LEVEL |

**Load limits.** The Caddy action's `ARGUS_MAX_INFLIGHT` (default 16) must be
less than or equal to `ARGUS_MAX_CONCURRENT_RPCS` (default 16). Requests above
the sidecar cap are rejected with RESOURCE_EXHAUSTED, which the action counts
as `overload` and fails open without tripping its circuit breaker. If you
raise one, raise the other. Inference errors return INTERNAL. CRS keeps
enforcing its rules either way. A deadline limits client waiting, but cannot
forcibly stop an ONNX call already running. The runtime has one shared
model/session with spinning disabled. Measure capacity and end-to-end latency
on the deployment VM before enabling block.

## Development and tests

`proto/argus.proto` is the wire contract shared with coraza-mlcheck
(`proto/argusv1/argus.proto` there). Regenerate Python stubs with
`scripts/gen-proto.sh` after changing it (use `grpcio-tools==1.84.0`). The
Docker build generates stubs itself. Commit a compatible proto change in both
repositories together.

```sh
python3 -m unittest discover -s tests -p test_compose.py -v
docker build -t argus-sidecar:test .
docker run --rm --entrypoint python -e PYTHONPATH=/app \
  -v "$PWD/tests:/tests:ro" argus-sidecar:test \
  -m unittest discover -s /tests -p test_runtime.py -v
# Uses the model path from .env.
docker compose run --rm --no-deps --entrypoint python -e PYTHONPATH=/app \
  -v "$PWD/tests:/tests:ro" argus-sidecar /tests/smoke_test.py
```

CI renders Compose using a persisted `.env` and a clean shell, checks
nondefault runtime overrides, and runs the runtime tests (including hot
reload against a fake bundle) without model artifacts. Pushing a `v*` tag
publishes `ghcr.io/mobytefon/argus-sidecar:<tag>`. The local model smoke test
checks clean/SQLi/JSON-body/traversal samples; `tests/e2e_client.py`
exercises the running UDS service. Use `requirements.lock` for the tested
dependency set; regenerate and rerun model parity tests before dependency
upgrades.

Rollback of the integration: remove ML rules and apply the Caddy config before
removing the action from the Caddy binary. Sidecar health alone does not prove
Caddy is invoking the action; confirm an ML audit event and a no-error verdict
via a test request.
