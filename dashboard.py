#!/usr/bin/env python3
"""Streamlit dashboard for inspecting MNIST model predictions.

The dashboard reads MNIST IDX files from the Renku-mounted Zenodo connector and
loads model artifacts created by the non-interactive training jobs:

- models/mnist-mlp-model.npz
- models/mnist-small-cnn.pt
"""

from __future__ import annotations

import gzip
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from torch import nn

DEFAULT_DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"))
DEFAULT_PRETRAINED_MODEL_DIR = Path(
    os.environ.get("PRETRAINED_MODEL_DIR", "/home/renku/work/pretrained-model-artifacts/mnist-models")
)
DEFAULT_SESSION_MODEL_DIR = Path(os.environ.get("SESSION_MODEL_DIR", "/home/renku/work/dashboard-trained-models"))


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


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def safe_size(path: Path) -> str:
    try:
        return f"{path.stat().st_size / 1024 / 1024:.2f} MB"
    except OSError:
        return "unavailable"


def run_training_with_progress(data_dir: Path, model_dir: Path, numpy_epochs: int, torch_epochs: int, sync_wait: int) -> bool:
    """Run the artifact-preparation script and stream logs into Streamlit."""
    script_dir = Path(__file__).resolve().parent
    prepare_script = script_dir / "prepare_model_artifacts.py"
    if not prepare_script.exists():
        st.error(f"Could not find training helper: {prepare_script}")
        return False

    command = [
        sys.executable,
        str(prepare_script),
        "--data-dir",
        str(data_dir),
        "--artifact-dir",
        str(model_dir),
        "--numpy-epochs",
        str(numpy_epochs),
        "--torch-epochs",
        str(torch_epochs),
        "--sync-wait",
        str(sync_wait),
    ]

    stage = st.empty()
    progress = st.progress(0, text="Starting training...")
    log_box = st.empty()
    logs: list[str] = []
    total_units = max(numpy_epochs + torch_epochs + 2, 1)
    current_stage_offset = 0

    stage.info("Preparing model artifact directory and starting NumPy MLP training...")
    process = subprocess.Popen(
        command,
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        logs.append(line)
        if len(logs) > 160:
            logs = logs[-160:]
        log_box.code("\n".join(logs), language="text")

        if "train_torch_mnist.py" in line or "Loaded MNIST" in line and "SmallCNN" in line:
            current_stage_offset = numpy_epochs
            stage.info("NumPy model saved. Training PyTorch CNN...")
        elif "Wrote readiness marker" in line:
            progress.progress(min((numpy_epochs + torch_epochs + 1) / total_units, 0.98), text="Artifacts verified; writing readiness marker...")
            stage.info("Artifacts verified. Flushing writes to the output connector...")
        elif "Outputs written, waiting" in line:
            progress.progress(min((numpy_epochs + torch_epochs + 1) / total_units, 0.98), text="Waiting for connector sync...")
            stage.info("Waiting for rclone/connector sync so files persist after the session exits...")
        elif "Artifact preparation complete" in line:
            progress.progress(1.0, text="Training complete")
            stage.success("Training complete. Artifacts are ready.")
        else:
            match = re.search(r"epoch=(\d+)", line)
            if match:
                epoch = int(match.group(1))
                done_units = min(current_stage_offset + epoch, numpy_epochs + torch_epochs)
                label = "Training PyTorch CNN" if current_stage_offset else "Training NumPy MLP"
                progress.progress(min(done_units / total_units, 0.95), text=f"{label}: epoch {epoch}")

    return_code = process.wait()
    if return_code != 0:
        progress.progress(0, text="Training failed")
        stage.error(f"Training failed with exit code {return_code}. See logs above.")
        return False

    load_cnn.clear()
    load_test_data.clear()
    return True


def main() -> None:
    st.set_page_config(page_title="MNIST model dashboard", layout="wide")
    st.title("MNIST model inspection dashboard")
    st.caption("Compare the NumPy MLP and PyTorch CNN trained by Renku non-interactive jobs.")

    if "model_source" not in st.session_state:
        st.session_state.model_source = "Pretrained models (public, read-only)"

    with st.sidebar:
        st.header("Paths")
        data_dir = Path(st.text_input("MNIST data directory", str(DEFAULT_DATA_DIR)))
        pretrained_model_dir = Path(st.text_input("Pretrained model directory (read-only)", str(DEFAULT_PRETRAINED_MODEL_DIR)))
        session_model_dir = Path(st.text_input("Session training output directory", str(DEFAULT_SESSION_MODEL_DIR)))
        custom_model_dir_text = st.text_input("Custom model directory", "")

        model_source = st.radio(
            "Model source",
            [
                "Pretrained models (public, read-only)",
                "Models trained in this session",
                "Custom path",
            ],
            key="model_source",
        )
        if model_source == "Models trained in this session":
            model_dir = session_model_dir
        elif model_source == "Custom path" and custom_model_dir_text.strip():
            model_dir = Path(custom_model_dir_text.strip())
        else:
            model_dir = pretrained_model_dir

        artifacts = artifact_status(model_dir)
        available = [name for name, path in artifacts.items() if safe_exists(path)]
        missing = {name: path for name, path in artifacts.items() if not safe_exists(path)}
        st.subheader("Artifacts")
        st.caption(f"Reading models from `{model_dir}`")
        for name, path in artifacts.items():
            status = "✅" if safe_exists(path) else "❌"
            size = f" ({safe_size(path)})" if safe_exists(path) else ""
            st.write(status, name, f"`{path}`{size}")

        st.subheader("Bootstrap training")
        st.caption("Interactive retraining writes to the session output directory, not the public read-only pretrained connector.")
        numpy_epochs = st.number_input("NumPy MLP epochs", min_value=1, max_value=100, value=30, step=1)
        torch_epochs = st.number_input("PyTorch CNN epochs", min_value=1, max_value=50, value=5, step=1)
        sync_wait = st.number_input("Post-write sync wait (seconds)", min_value=0, max_value=600, value=0, step=30)
        train_button_label = "Train models in this session" if missing else "Retrain into session directory"
        train_requested = st.button(train_button_label, type="primary" if missing else "secondary")

    try:
        images, labels = load_test_data(str(data_dir))
    except Exception as exc:
        st.error(f"Could not load MNIST test data from {data_dir}: {exc}")
        st.stop()

    if train_requested:
        st.header("Training models in this session")
        st.write(
            "This runs the same training scripts as the non-interactive Renku job. In the dashboard it writes to "
            f"the session-local output directory `{session_model_dir}`, so users do not need write access to the "
            "public pretrained-model connector."
        )
        ok = run_training_with_progress(data_dir, session_model_dir, int(numpy_epochs), int(torch_epochs), int(sync_wait))
        if ok:
            st.session_state.model_source = "Models trained in this session"
            st.success("Models trained successfully. Switching dashboard to the session-trained models...")
            st.rerun()
        st.stop()

    if not available:
        st.warning("No model artifacts found for the selected model source. You can train models directly from this dashboard, or run the Renku training job launcher.")
        st.info("Use the **Train models in this session** button in the sidebar to bootstrap writable session-local artifacts and watch progress here.")
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
