# Renku non-interactive MNIST training job

This repository is a minimal example for running a non-interactive machine-learning training job on Renku.

It includes two training approaches on MNIST IDX files from Zenodo:

- `train_torch_mnist.py`: a PyTorch CNN, used by the Renku non-interactive job.
- `train_mnist.py`: a dependency-light NumPy MLP baseline.

- DOI: [`10.5281/zenodo.10058130`](https://doi.org/10.5281/zenodo.10058130)

The intended Renku workflow is:

1. Link this code repository to a Renku project.
2. Link/create a Zenodo data connector for the MNIST record, mounted at `data/`.
3. Build a Python environment from `requirements.txt`.
4. Create a non-interactive launcher that runs:

   ```bash
   python train_torch_mnist.py \
     --data-dir /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130 \
     --output-dir outputs \
     --epochs 5
   ```

5. Launch it as a Renku non-interactive job.

## Local test

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python train_torch_mnist.py --download-if-missing --epochs 1 --train-limit 5000 --test-limit 1000
```

The script writes:

- `outputs/metrics.json`
- `outputs/mnist-small-cnn.pt`

The NumPy baseline writes `outputs/mnist-mlp-model.npz`.

The final log line starts with `FINAL_METRICS` and contains JSON metrics, which makes job logs easy to parse.
