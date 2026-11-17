# Argus Sentinel sidecar

Python/ONNX CPU inference service for the Coraza `mlcheck` action in
[mobytefon/coraza-mlcheck](https://github.com/mobytefon/coraza-mlcheck).
The action calls it over a Unix socket (`/run/argus/argus.sock`) on a shared
Docker volume. The classifier is binary: `attack_prob > threshold`; the gRPC
response uses `risk_score`, `is_clean` and an optional `attack` label. It does
not label SQLi/XSS.

The released model bundle (Argus Sentinel v4, run `D_c64`) is in
[`model/`](model/): `model.onnx`, `tokenizer.json`, `config.json`,
`preprocess.py` and `inference_cpu.py`. The image contains only the code; the
model directory is mounted read-only at `/model`. The two Python files are
executed by the sidecar: only mount bundles from a source you trust.

## With MOBI-WAF

[MOBI-WAF](https://github.com/mobytefon/MOBI-WAF) runs this image from its
own `docker-compose.argus.yml`, as `0:0` like its `web` and `caddy` services
(rootless Docker), on the `argus-run` volume its Caddy mounts. Follow its
`documentation/argus-ml.md`; this repository's `compose.yaml` is not needed
there.

## Standalone (another Caddy with the mlcheck action)

Requirements: Docker Engine with Compose v2, Linux amd64.

```sh
git clone --branch v0.2.1 https://github.com/mobytefon/argus-sidecar.git
cd argus-sidecar
cp .env.example .env      # ARGUS_MODEL_DIR_HOST=./model uses the bundled model
docker compose config --quiet
docker compose pull       # or: docker compose build --pull
docker compose up -d
docker compose exec argus-sidecar python -m argus_sidecar.healthcheck
```

Mount the same named volume (`ARGUS_SOCKET_VOLUME`, default `argus-run`) at
`/run/argus` in Caddy. Keep `.env` in the checkout: Compose reads it again
after a new SSH login. The model path is a **host path**; the container always
reads `/model`. For a deployment, copy the five files to a directory of their
own and point `ARGUS_MODEL_DIR_HOST` at it, so a model swap does not touch the
checkout.

### Users, permissions and rootless Docker

| | Image user `10000:10000` (default) | `ARGUS_USER=0:0` (rootless Docker) |
|---|---|---|
| Who it is on the host | an unprivileged UID (rootful) or a sub-UID (rootless) | rootless: the user running `dockerd`; rootful: real root, with every capability dropped |
| Model directory | readable by UID 10000 (`chmod -R a+rX`, or `chown -R 10000:10000`) | rootless: any directory owned by that user, `0700`/`0600` is fine; rootful (Docker Desktop too): root without capabilities cannot read other users' private files, so `chmod 0755` the directory and `0644` the files, or `chown -R root` |
| Socket | `srw-rw----` 10000:10000: Caddy must run as 10000, or as root | `srw-rw----` root: Caddy must run as root (`0:0`) |

The container drops every capability, so the sidecar relies on file
ownership alone. `/run/argus` in the image is mode `1777`, so a fresh socket
volume can be used by either user. Run the sidecar as the same user as Caddy
(`0:0` for both under rootless Docker). After switching user on an existing
volume, a socket left by the other user cannot be replaced: stop both
services and `docker volume rm` the socket volume once.

One sidecar per Caddy node; do not share one socket filesystem across hosts.
A lock prevents two sidecars from unlinking each other's live socket. The
container runs with `network_mode: none`: it only talks over the socket.

## Hot model swap

The sidecar watches the five bundle files and swaps the model without a
restart and without dropping requests:

1. Copy **all five** new files over the model directory, in one command:

   ```sh
   cp new/model.onnx new/tokenizer.json new/config.json new/preprocess.py new/inference_cpu.py ~/argus-model/
   ```

2. Watch the log: `docker compose logs -f argus-sidecar`. About two
   `ARGUS_RELOAD_INTERVAL_S` periods after the last file stops changing you
   see `model reloaded version=… run=… threshold=…`. To reload immediately:
   `docker compose kill -s HUP argus-sidecar`.

How it behaves:

- Every `ARGUS_RELOAD_INTERVAL_S` seconds (default 5; `0` disables polling)
  the sidecar compares the files' modification time and size with the running
  model's. A change is loaded only once the files are identical across two
  consecutive checks, so a copy in progress is not picked up half-written.
  Copying the files one by one, more than one interval apart, can briefly load
  a mix of old and new files: copy them together.
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
  files are not retried until they change again (or on SIGHUP).
- Requests already running finish on the model they started with.
- A broken bundle at **startup** still exits the process (Compose restarts it).

Rollback: copy the previous bundle back the same way. Keep it around.

## Configuration

| Environment | Default | Meaning |
|---|---|---|
| ARGUS_MODEL_DIR_HOST | `./model` in `.env.example` | Host bundle path; mounted read-only at /model |
| ARGUS_USER | 10000:10000 | Container user; `0:0` under rootless Docker |
| ARGUS_SOCKET_VOLUME | argus-run | Shared Docker volume name |
| ARGUS_THRESHOLD | fpr0.1% | Preset from the model's config.json, or a finite float 0..1 |
| ARGUS_RELOAD_INTERVAL_S | 5 | Seconds between model-file checks, 0..3600; 0 disables polling (SIGHUP still reloads) |
| ARGUS_SOCKET | /run/argus/argus.sock | Absolute Unix socket path, inside /run/argus |
| ARGUS_SOCKET_MODE | 0o660 | 0o600 or 0o660 |
| ARGUS_WORKERS | 4 | Inference worker threads, 1..256 |
| ARGUS_MAX_CONCURRENT_RPCS | 16 | Requests served at once, 1..1024; more are refused as busy |
| ARGUS_MIN_REMAINING_MS | 5 | Reject requests with almost-expired deadlines |
| ORT_INTRA_THREADS | 1 | ONNX threads per session |
| ARGUS_LOG_LEVEL | INFO | Forwarded to the runtime as LOG_LEVEL |
| ARGUS_CPUS / ARGUS_MEM_LIMIT | 2 / 512m | Container limits (rootless Docker enforces them only with cgroup v2 delegation) |

**Load limits.** The Caddy action's `ARGUS_MAX_INFLIGHT` (default 16) must be
less than or equal to `ARGUS_MAX_CONCURRENT_RPCS` (default 16). Requests above
the sidecar cap are rejected with RESOURCE_EXHAUSTED, which the action counts
as `overload` and fails open without tripping its circuit breaker. If you
raise one, raise the other, with `ARGUS_WORKERS` and `ARGUS_CPUS`. Inference
errors return INTERNAL. A deadline limits client waiting, but cannot stop an
ONNX call already running. Measure capacity and end-to-end latency on the
deployment host before enabling block.

**Keepalive.** The action keeps one gRPC connection open and pings it every
30 s, also when idle. The server accepts pings down to 10 s apart; with
gRPC's default policy (one idle ping per 5 minutes) it would close Caddy's
connection with GOAWAY `too_many_pings` after a few idle minutes.

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
docker run --rm --entrypoint python -e PYTHONPATH=/app \
  -v "$PWD/tests:/tests:ro" -v "$PWD/model:/model:ro" argus-sidecar:test \
  /tests/smoke_test.py --model-dir /model
```

CI renders Compose from a persisted `.env` in a clean shell, runs the runtime
tests (hot reload against a fake bundle), the smoke test with the bundled
model, and serves the bundled model as `0:0` with every capability dropped
from a root-only model directory, called from a second container. Pushing a
`v*` tag publishes `ghcr.io/mobytefon/argus-sidecar:<tag>`. Use
`requirements.lock` for the tested dependency set.

Rollback of the integration: remove the ML rules and apply the Caddy config
before removing the action from the Caddy binary. Sidecar health alone does
not prove Caddy is invoking the action; confirm an ML audit event and a
no-error verdict via a test request.
