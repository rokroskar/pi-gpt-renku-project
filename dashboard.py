#!/usr/bin/env python3
"""Streamlit dashboard for inspecting MNIST model predictions.

The dashboard reads MNIST IDX files from the Renku-mounted Zenodo connector and
loads model artifacts created by the non-interactive training jobs:

- models/mnist-mlp-model.npz
- models/mnist-small-cnn.pt
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from torch import nn

DEFAULT_DATA_DIR = Path("/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130")
DEFAULT_MODEL_DIR = Path("/home/renku/work/models")


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def open_idx(path: Path):
    if path.exists():
        return path.open("rb")
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(f"Could not find {path} or {gz_path}")


@st.cache_data(show_spinner="Loading MNIST test set")
def load_test_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_dir)
    with open_idx(root / "t10k-images-idx3-ubyte") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number: {magic}")
        images = np.frombuffer(f.read(count * rows * cols), dtype=np.uint8).reshape(count, rows, cols)
    with open_idx(root / "t10k-labels-idx1-ubyte") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number: {magic}")
        labels = np.frombuffer(f.read(count), dtype=np.uint8)
    return images, labels


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def mlp_predict(model_path: Path, image: np.ndarray) -> tuple[int, np.ndarray]:
    params = np.load(model_path)
    x = image.reshape(1, 784).astype(np.float32) / 255.0
    h1 = np.maximum(x @ params["w1"] + params["b1"], 0.0)
    h2 = np.maximum(h1 @ params["w2"] + params["b2"], 0.0)
    logits = h2 @ params["w3"] + params["b3"]
    probs = softmax(logits)[0]
    return int(probs.argmax()), probs


@st.cache_resource(show_spinner="Loading PyTorch CNN")
def load_cnn(model_path: str) -> SmallCNN:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = SmallCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def cnn_predict(model_path: Path, image: np.ndarray) -> tuple[int, np.ndarray]:
    model = load_cnn(str(model_path))
    x = torch.from_numpy(image.reshape(1, 1, 28, 28).astype(np.float32) / 255.0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def artifact_status(model_dir: Path) -> dict[str, Path]:
    return {
        "NumPy MLP": model_dir / "mnist-mlp-model.npz",
        "PyTorch CNN": model_dir / "mnist-small-cnn.pt",
    }


def main() -> None:
    st.set_page_config(page_title="MNIST model dashboard", layout="wide")
    st.title("MNIST model inspection dashboard")
    st.caption("Compare the NumPy MLP and PyTorch CNN trained by Renku non-interactive jobs.")

    with st.sidebar:
        st.header("Paths")
        data_dir = Path(st.text_input("MNIST data directory", str(DEFAULT_DATA_DIR)))
        model_dir = Path(st.text_input("Model artifact directory", str(DEFAULT_MODEL_DIR)))
        artifacts = artifact_status(model_dir)
        available = [name for name, path in artifacts.items() if path.exists()]
        missing = {name: path for name, path in artifacts.items() if not path.exists()}
        st.subheader("Artifacts")
        for name, path in artifacts.items():
            st.write(("✅" if path.exists() else "❌"), name, f"`{path}`")

    try:
        images, labels = load_test_data(str(data_dir))
    except Exception as exc:
        st.error(f"Could not load MNIST test data from {data_dir}: {exc}")
        st.stop()

    if not available:
        st.warning("No model artifacts found. Run the training job(s) first so `models/` contains saved models.")
        st.code(
            "python train_mnist.py --data-dir /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130 --model-dir models --epochs 30\n"
            "python train_torch_mnist.py --data-dir /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130 --model-dir models --epochs 5",
            language="bash",
        )
        st.stop()

    col_controls, col_image, col_pred = st.columns([1.2, 1, 1.5])
    with col_controls:
        model_name = st.selectbox("Model", available)
        idx = st.slider("Test example index", min_value=0, max_value=len(images) - 1, value=0)
        if st.button("Random example"):
            idx = int(np.random.default_rng().integers(0, len(images)))
        image = images[idx]
        label = int(labels[idx])
        st.write(f"Selected test example: `{idx}`")
        st.write(f"True label: **{label}**")

    with col_image:
        st.subheader("Digit")
        st.image(image, caption=f"MNIST test #{idx}", width=220, clamp=True)

    with col_pred:
        st.subheader("Prediction")
        try:
            if model_name == "NumPy MLP":
                pred, probs = mlp_predict(artifacts[model_name], image)
            else:
                pred, probs = cnn_predict(artifacts[model_name], image)
        except Exception as exc:
            st.error(f"Could not run prediction with {model_name}: {exc}")
            st.stop()
        st.metric("Predicted digit", pred, delta="correct" if pred == label else "incorrect")
        st.bar_chart({str(i): float(probs[i]) for i in range(10)})

    if missing:
        with st.expander("Missing models"):
            for name, path in missing.items():
                st.write(f"- {name}: `{path}`")


if __name__ == "__main__":
    main()
