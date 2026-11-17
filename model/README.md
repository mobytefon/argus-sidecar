Private runtime artifacts belong here, or in an operator-managed directory.
No model weights, tokenizer, training data or model config are published.

Copy the five files of the Argus Sentinel v4 bundle here (or into the
directory named by `ARGUS_MODEL_DIR_HOST`): `model.onnx`, `tokenizer.json`,
`config.json`, `preprocess.py` and `inference_cpu.py`, readable by UID 10000.
Mount the directory at `/model:ro`. Copying new files over a running
sidecar's directory hot-reloads the model (see the main README).

Only use trusted artifacts supplied by the operator: the two Python files are
executed by the sidecar.
