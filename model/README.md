# Argus Sentinel v4 model bundle

The five files the sidecar loads, as trained and exported (run `D_c64`,
`version: v4-binary`, 332,865 parameters):

| File | What |
|---|---|
| `model.onnx` | CNN classifier, ONNX opset 17. Inputs `input_ids` / `attention_mask` (int64, `[batch, seq]`, seq ≤ 256), output `attack_prob` (float32, `[batch, 1]`, sigmoid) |
| `tokenizer.json` | Byte-level BPE, vocabulary 4,096 |
| `config.json` | Input format, field caps, threshold presets (`fpr0.1%` 0.998175, `fpr0.5%` 0.893281, `fpr1%` 0.59351, `0.5`) and evaluation metrics |
| `preprocess.py` | Request normalization shared by training and inference |
| `inference_cpu.py` | `ArgusSentinelV4`, ONNX Runtime CPU inference |

The image does not contain the model: mount a directory holding these five
files at `/model` (read-only). For a deployment, copy them to a directory of
their own rather than mounting this checkout:

```sh
install -d ~/argus-model
cp model/model.onnx model/tokenizer.json model/config.json model/preprocess.py model/inference_cpu.py ~/argus-model/
```

Replacing the files in that directory while the sidecar runs switches it to
the new model (see the main README). A retrained model must ship all five
files together, with the thresholds recalibrated in its `config.json`.

`preprocess.py` and `inference_cpu.py` are executed by the sidecar. Only
mount bundles from a source you trust.

Measured on the held-out test set (3,592 requests, same distribution as the
training data, CSIC 2010 plus a synthetic set): ROC-AUC 0.9895; detection rate
86.3% / 89.7% / 91.8% at 0.1% / 0.5% / 1% false positives. Real traffic
differs: run detect-only first and recalibrate the threshold on it.
