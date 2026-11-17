"""Check the actual Compose renderer without Docker containers or model files.

Run on the host: python -m unittest discover -s tests -p test_compose.py -v
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComposeTests(unittest.TestCase):
    def render(self, overrides=""):
        with tempfile.TemporaryDirectory(prefix="argus-compose-") as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "compose.yaml", root / "compose.yaml")
            # No exported Argus/Compose values: simulate a new login shell.
            env = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("ARGUS_", "ORT_", "COMPOSE_"))}
            (root / ".env").write_text(
                (ROOT / ".env.example").read_text(encoding="utf-8") + "\n" + overrides,
                encoding="utf-8",
            )
            result = subprocess.run(
                ["docker", "compose", "--project-directory", str(root),
                 "-f", str(root / "compose.yaml"), "config", "--format", "json"],
                cwd=root, env=env, capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)

    def test_persisted_example_works_without_shell_exports(self):
        config = self.render()
        service = config["services"]["argus-sidecar"]
        self.assertEqual(service["image"], "ghcr.io/mobytefon/argus-sidecar:v0.2.0")
        self.assertEqual(service["network_mode"], "none")
        model = next(v for v in service["volumes"] if v["target"] == "/model")
        self.assertTrue(model["source"].replace("\\", "/").endswith("/opt/argus/models/v4"))
        self.assertTrue(model["read_only"])
        self.assertEqual(config["volumes"]["argus-run"]["name"], "ingressi-argus-run")
        env = service["environment"]
        self.assertEqual(env["ARGUS_SOCKET"], "/run/argus/argus.sock")
        self.assertEqual(env["ARGUS_WORKERS"], "4")
        self.assertEqual(env["ARGUS_MAX_CONCURRENT_RPCS"], "16")
        self.assertEqual(env["ARGUS_RELOAD_INTERVAL_S"], "5")
        self.assertEqual(env["LOG_LEVEL"], "INFO")
        self.assertNotIn("ARGUS_VERIFY_MANIFEST", env)

    def test_compose_defaults_match_the_example(self):
        # Removing a value from .env must not change the effective default.
        with tempfile.TemporaryDirectory(prefix="argus-compose-") as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "compose.yaml", root / "compose.yaml")
            (root / ".env").write_text("ARGUS_MODEL_DIR_HOST=/opt/argus/models/v4\n", encoding="utf-8")
            env = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith(("ARGUS_", "ORT_", "COMPOSE_"))}
            result = subprocess.run(
                ["docker", "compose", "--project-directory", str(root),
                 "-f", str(root / "compose.yaml"), "config", "--format", "json"],
                cwd=root, env=env, capture_output=True, text=True, check=True,
            )
        bare = json.loads(result.stdout)["services"]["argus-sidecar"]["environment"]
        example = self.render()["services"]["argus-sidecar"]["environment"]
        self.assertEqual(bare, example)

    def test_dockerfile_defaults_match_compose(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        env_line = next(line for line in dockerfile.splitlines() if line.startswith("ENV "))
        image_env = dict(re.findall(r"(\w+)=(\S+)", env_line))
        compose_env = self.render()["services"]["argus-sidecar"]["environment"]
        for name in ("ARGUS_WORKERS", "ARGUS_MAX_CONCURRENT_RPCS", "ARGUS_RELOAD_INTERVAL_S",
                     "ARGUS_THRESHOLD", "ARGUS_MIN_REMAINING_MS", "ARGUS_SOCKET"):
            with self.subTest(variable=name):
                self.assertEqual(image_env[name], compose_env[name])
        self.assertNotIn("ARGUS_VERIFY_MANIFEST", image_env)

    def test_nondefault_runtime_settings_reach_the_container(self):
        custom = {
            "ARGUS_SOCKET": "/run/argus/custom.sock", "ARGUS_SOCKET_MODE": "0o600",
            "ARGUS_THRESHOLD": "0.95", "ARGUS_RELOAD_INTERVAL_S": "0",
            "ARGUS_WORKERS": "2", "ARGUS_MAX_CONCURRENT_RPCS": "4",
            "ARGUS_MIN_REMAINING_MS": "20", "ORT_INTRA_THREADS": "2",
            "ARGUS_LOG_LEVEL": "DEBUG", "ARGUS_SOCKET_VOLUME": "argus-custom-volume",
        }
        config = self.render("\n".join(f"{k}={v}" for k, v in custom.items()) + "\n")
        actual = config["services"]["argus-sidecar"]["environment"]
        for name, value in custom.items():
            if name == "ARGUS_SOCKET_VOLUME":
                self.assertEqual(config["volumes"]["argus-run"]["name"], value)
            else:
                target = "LOG_LEVEL" if name == "ARGUS_LOG_LEVEL" else name
                with self.subTest(variable=name):
                    self.assertEqual(actual[target], value)
        self.assertEqual(actual["ARGUS_MODEL_DIR"], "/model")


if __name__ == "__main__":
    unittest.main()
