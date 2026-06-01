# MNIST Gradio dashboard

This project includes an interactive Gradio dashboard for inspecting MNIST test-set predictions from model artifacts produced by Renku jobs.

## Launcher

Use the Renku launcher named **MNIST Gradio model dashboard**.

The launcher runs `dashboard.py` from the runtime-cloned GitHub repository and serves Gradio under the Renku session path prefix.

## Default paths

The dashboard uses separate paths for read-only pre-trained artifacts and writable interactive retraining output:

- MNIST data: `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`
- Public read-only pre-trained artifacts: `/home/renku/work/pretrained-model-artifacts/mnist-models`
- Session-local retraining output: `/home/renku/work/dashboard-trained-models`

The non-interactive training job still writes to the private writable output connector at `/home/renku/work/model-artifacts/mnist-models`; the dashboard reads pre-trained public artifacts from the separate read-only connector so users without write permission can still use the demo.

Expected artifact files:

- `mnist-mlp-model.npz` — NumPy MLP
- `mnist-small-cnn.pt` — PyTorch CNN
- `_ARTIFACTS_READY.json` — readiness marker written after artifact verification

## What the dashboard shows

The dashboard lets users:

- choose between the NumPy MLP and PyTorch CNN models;
- select or randomize an MNIST test example;
- view the digit image and true label;
- inspect the predicted digit;
- compare class probabilities in a bar chart;
- see whether model artifacts are present and their sizes.

## Bootstrap training from the dashboard

If the pre-trained artifacts are missing or unreadable, the dashboard can train models directly from the interactive session. These dashboard-triggered runs write to `/home/renku/work/dashboard-trained-models`, not to the public read-only connector.

Use the sidebar button:

- **Train models in this session** when artifacts are absent;
- **Retrain into session directory** when artifacts already exist.

The dashboard then runs `prepare_model_artifacts.py`, which:

1. trains the NumPy MLP;
2. trains the PyTorch CNN;
3. verifies both artifact files;
4. writes `_ARTIFACTS_READY.json`;
5. calls `os.sync()`;
6. optionally waits after flushing writes.

During this process, the dashboard shows a progress bar, current training stage, and streamed logs.

The preferred production workflow is still the non-interactive **Prepare MNIST model artifacts job** launcher. The dashboard bootstrap option is intended as a recovery/demo path so a user can still see the project run if public pre-trained artifacts are missing or they want to retrain without write access to the artifact connector.

## Local run

For local development after installing dependencies:

```bash
python dashboard.py
```

Set custom paths with environment variables if needed:

```bash
MNIST_DATA_DIR=/path/to/mnist PRETRAINED_MODEL_DIR=/path/to/pretrained SESSION_MODEL_DIR=/path/to/session-models python dashboard.py
```
