# MNIST Streamlit dashboard

This project includes an interactive Streamlit dashboard for inspecting MNIST test-set predictions from model artifacts produced by Renku jobs.

## Launcher

Use the Renku launcher named **MNIST Streamlit model dashboard**.

The launcher runs `dashboard.py` from the runtime-cloned GitHub repository and serves Streamlit under the Renku session path prefix.

## Default paths

The dashboard defaults to the Renku connector mounts used by this project:

- MNIST data: `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`
- Model artifacts: `/home/renku/work/model-artifacts/mnist-models`

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

If the saved artifacts are missing or unreadable, the dashboard can train them directly from the interactive session.

Use the sidebar button:

- **Train missing models** when artifacts are absent;
- **Retrain / overwrite models** when artifacts already exist.

The dashboard then runs `prepare_model_artifacts.py`, which:

1. trains the NumPy MLP;
2. trains the PyTorch CNN;
3. verifies both artifact files;
4. writes `_ARTIFACTS_READY.json`;
5. calls `os.sync()`;
6. waits for the mounted output connector to flush writes.

During this process, the dashboard shows a progress bar, current training stage, and streamed logs.

The preferred production workflow is still the non-interactive **Prepare MNIST model artifacts job** launcher. The dashboard bootstrap option is intended as a recovery/demo path so a user can still see the project run if artifacts are missing.

## Local run

For local development after installing dependencies:

```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8080
```

Set custom paths with environment variables if needed:

```bash
MNIST_DATA_DIR=/path/to/mnist MODEL_DIR=/path/to/models streamlit run dashboard.py
```
