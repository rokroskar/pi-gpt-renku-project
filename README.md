# Renku non-interactive MNIST training job

This repository demonstrates a Renku project with **non-interactive machine-learning training jobs** and an **interactive Streamlit dashboard** for inspecting saved model artifacts.

The project trains two MNIST classifiers from IDX files provided by Zenodo DOI [`10.5281/zenodo.10058130`](https://doi.org/10.5281/zenodo.10058130):

- `train_mnist.py`: dependency-light NumPy MLP baseline.
- `train_torch_mnist.py`: PyTorch CNN with higher accuracy.

In Renku, the MNIST data is mounted from the Zenodo data connector. The jobs do **not** download MNIST at runtime.

## Renku workflow

The intended Renku setup uses three launchers:

1. **Build Python environment from repository**
   - Builds the Python environment from `requirements.txt`.
   - A rebuild is only needed when dependencies change.

2. **Prepare MNIST model artifacts job**
   - A non-interactive job launcher.
   - Runs `prepare_model_artifacts.py`, which trains both models and writes artifacts to a writable output connector.
   - Includes `os.sync()` and a configurable wait so rclone-backed output connectors have time to flush writes before the job exits.

3. **MNIST Streamlit model dashboard**
   - An interactive Streamlit session.
   - Loads the mounted MNIST test set and saved model artifacts.
   - Lets users switch between models, select/randomize test examples, and inspect prediction probabilities.
   - Can also bootstrap/retrain missing models directly from the dashboard with live progress and logs.

## Renku paths

Default mounted paths used by the scripts and dashboard:

- MNIST data connector:
  `/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`
- Writable model artifact connector:
  `/home/renku/work/model-artifacts`
- Model artifact directory:
  `/home/renku/work/model-artifacts/mnist-models`

Expected artifacts:

- `/home/renku/work/model-artifacts/mnist-models/mnist-mlp-model.npz`
- `/home/renku/work/model-artifacts/mnist-models/mnist-small-cnn.pt`
- `/home/renku/work/model-artifacts/mnist-models/_ARTIFACTS_READY.json`

## Running the artifact job manually

The non-interactive launcher runs the equivalent of:

```bash
python prepare_model_artifacts.py \
  --data-dir /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130 \
  --artifact-dir /home/renku/work/model-artifacts/mnist-models \
  --numpy-epochs 30 \
  --torch-epochs 5 \
  --sync-wait 180
```

The final training logs include `FINAL_METRICS` JSON records from both trainers. The preparation script verifies the artifacts, writes `_ARTIFACTS_READY.json`, calls `os.sync()`, and waits for connector synchronization.

## Dashboard

The dashboard is implemented in `dashboard.py` and runs with Streamlit:

```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8080
```

Inside Renku, the dashboard launcher uses the runtime-cloned repository path and the Renku path prefix, for example:

```bash
streamlit run /home/renku/work/pi-gpt-renku-project/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port 8080 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.baseUrlPath ${RENKU_BASE_URL_PATH#/}
```

The sidebar shows artifact availability and sizes. If artifacts are missing or corrupted, use **Train missing models** to run the same preparation workflow interactively with progress bars and streamed logs.

See [`docs/dashboard.md`](docs/dashboard.md) for more dashboard details.

## Local test

For a small local smoke test that downloads MNIST only for local development:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python train_torch_mnist.py --download-if-missing --epochs 1 --train-limit 5000 --test-limit 1000
```

For Renku jobs, use the mounted Zenodo connector instead of `--download-if-missing`.
