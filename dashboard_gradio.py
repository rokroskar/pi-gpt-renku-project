#!/usr/bin/env python3
"""Gradio dashboard for inspecting MNIST model predictions.

The dashboard reads MNIST IDX files from the Renku-mounted Zenodo connector and
loads model artifacts created by the non-interactive training job. By default it
reads public, read-only pre-trained artifacts. If a user retrains from the UI,
new artifacts are written to a session-local writable directory.
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import re
import struct
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Generator

import gradio as gr
import numpy as np
import torch
from torch import nn

DEFAULT_DATA_DIR = Path(os.environ.get("MNIST_DATA_DIR", "/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130"))
DEFAULT_PRETRAINED_MODEL_DIR = Path(
    os.environ.get("PRETRAINED_MODEL_DIR", "/home/renku/work/pretrained-model-artifacts/mnist-models")
)
DEFAULT_SESSION_MODEL_DIR = Path(os.environ.get("SESSION_MODEL_DIR", "/home/renku/work/dashboard-trained-models"))
MODEL_NAMES = ["NumPy MLP", "PyTorch CNN"]


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


@lru_cache(maxsize=8)
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


def artifact_paths(model_dir: Path) -> dict[str, Path]:
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


def selected_model_dir(source: str, pretrained_dir: str, session_dir: str, custom_dir: str) -> Path:
    if source == "Models trained in this session":
        return Path(session_dir)
    if source == "Custom path" and custom_dir.strip():
        return Path(custom_dir.strip())
    return Path(pretrained_dir)


def artifact_status(source: str, pretrained_dir: str, session_dir: str, custom_dir: str) -> tuple[str, gr.Dropdown]:
    model_dir = selected_model_dir(source, pretrained_dir, session_dir, custom_dir)
    paths = artifact_paths(model_dir)
    lines = [f"Reading models from `{model_dir}`", ""]
    available = []
    for name, path in paths.items():
        if safe_exists(path):
            available.append(name)
            lines.append(f"✅ **{name}** — `{path}` ({safe_size(path)})")
        else:
            lines.append(f"❌ **{name}** — `{path}`")
    if not available:
        lines.append("")
        lines.append("No artifacts found for this model source. Use **Train models in this session** below.")
    return "\n".join(lines), gr.Dropdown(choices=available or MODEL_NAMES, value=(available[0] if available else MODEL_NAMES[0]))


def mlp_predict(model_path: Path, image: np.ndarray) -> tuple[int, np.ndarray]:
    params = np.load(model_path)
    x = image.reshape(1, 784).astype(np.float32) / 255.0
    h1 = np.maximum(x @ params["w1"] + params["b1"], 0.0)
    h2 = np.maximum(h1 @ params["w2"] + params["b2"], 0.0)
    logits = h2 @ params["w3"] + params["b3"]
    probs = softmax(logits)[0]
    return int(probs.argmax()), probs


@lru_cache(maxsize=8)
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


def predict(
    data_dir: str,
    source: str,
    pretrained_dir: str,
    session_dir: str,
    custom_dir: str,
    model_name: str,
    index: int,
) -> tuple[np.ndarray | None, str, dict[str, float]]:
    try:
        images, labels = load_test_data(data_dir)
        index = max(0, min(int(index), len(images) - 1))
        image = images[index]
        label = int(labels[index])
        model_dir = selected_model_dir(source, pretrained_dir, session_dir, custom_dir)
        paths = artifact_paths(model_dir)
        model_path = paths[model_name]
        if not safe_exists(model_path):
            return image, f"Missing artifact for {model_name}: {model_path}", {}
        if model_name == "NumPy MLP":
            pred, probs = mlp_predict(model_path, image)
        else:
            pred, probs = cnn_predict(model_path, image)
        status = f"Test example #{index} · true label: {label} · predicted: {pred} ({'correct' if pred == label else 'incorrect'})"
        return image, status, {str(i): float(probs[i]) for i in range(10)}
    except Exception as exc:
        return None, f"Error: {exc}", {}


def random_index(data_dir: str) -> int:
    try:
        images, _ = load_test_data(data_dir)
        return random.randint(0, len(images) - 1)
    except Exception:
        return 0


def run_training_with_progress(
    data_dir: str,
    session_dir: str,
    numpy_epochs: int,
    torch_epochs: int,
    sync_wait: int,
) -> Generator[tuple[str, str], None, None]:
    """Run the artifact-preparation script and stream progress/logs to Gradio."""
    script_dir = Path(__file__).resolve().parent
    prepare_script = script_dir / "prepare_model_artifacts.py"
    if not prepare_script.exists():
        yield f"Could not find training helper: {prepare_script}", ""
        return

    command = [
        sys.executable,
        str(prepare_script),
        "--data-dir",
        data_dir,
        "--artifact-dir",
        session_dir,
        "--numpy-epochs",
        str(int(numpy_epochs)),
        "--torch-epochs",
        str(int(torch_epochs)),
        "--sync-wait",
        str(int(sync_wait)),
    ]

    logs: list[str] = []
    status = "Starting training in the session-local output directory..."
    yield status, ""

    process = subprocess.Popen(
        command,
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    current_stage = "NumPy MLP"
    for line in process.stdout:
        line = line.rstrip()
        logs.append(line)
        if len(logs) > 180:
            logs = logs[-180:]

        if "train_torch_mnist.py" in line or ("Loaded MNIST" in line and "SmallCNN" in line):
            current_stage = "PyTorch CNN"
            status = "NumPy model saved. Training PyTorch CNN..."
        elif "Wrote readiness marker" in line:
            status = "Artifacts verified. Writing readiness marker..."
        elif "Outputs written, waiting" in line:
            status = "Outputs written. Waiting after flush/sync..."
        elif "Artifact preparation complete" in line:
            status = "Training complete. Session-trained artifacts are ready. Select 'Models trained in this session' as the model source."
        else:
            match = re.search(r"epoch=(\d+)", line)
            if match:
                status = f"Training {current_stage}: epoch {match.group(1)}"

        yield status, "\n".join(logs)

    return_code = process.wait()
    load_cnn.cache_clear()
    if return_code == 0:
        yield "Training complete. Session-trained artifacts are ready. Select 'Models trained in this session' as the model source.", "\n".join(logs)
    else:
        yield f"Training failed with exit code {return_code}. See logs below.", "\n".join(logs)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MNIST model dashboard") as app:
        gr.Markdown(
            "# MNIST model inspection dashboard\n"
            "Compare the NumPy MLP and PyTorch CNN trained by Renku non-interactive jobs. "
            "The default model source is the public read-only pretrained-artifacts connector."
        )

        with gr.Row():
            with gr.Column(scale=1):
                data_dir = gr.Textbox(label="MNIST data directory", value=str(DEFAULT_DATA_DIR))
                pretrained_dir = gr.Textbox(label="Pretrained model directory (read-only)", value=str(DEFAULT_PRETRAINED_MODEL_DIR))
                session_dir = gr.Textbox(label="Session training output directory", value=str(DEFAULT_SESSION_MODEL_DIR))
                custom_dir = gr.Textbox(label="Custom model directory", value="")
                source = gr.Radio(
                    label="Model source",
                    choices=["Pretrained models (public, read-only)", "Models trained in this session", "Custom path"],
                    value="Pretrained models (public, read-only)",
                )
                refresh = gr.Button("Refresh artifact status")
                artifact_md = gr.Markdown("Click **Refresh artifact status** to check available model artifacts.")
                model_name = gr.Dropdown(label="Model", choices=MODEL_NAMES, value=MODEL_NAMES[0])
                index = gr.Slider(label="Test example index", minimum=0, maximum=9999, step=1, value=0)
                random_btn = gr.Button("Random example")
                predict_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                image = gr.Image(label="Digit", type="numpy", height=260)
                prediction = gr.Textbox(label="Prediction", lines=3)
                probs = gr.Label(label="Class probabilities", num_top_classes=10)

        with gr.Accordion("Train models in this session", open=False):
            gr.Markdown(
                "Interactive retraining writes to the session output directory, not the public read-only pretrained connector."
            )
            with gr.Row():
                numpy_epochs = gr.Number(label="NumPy MLP epochs", value=30, precision=0, minimum=1)
                torch_epochs = gr.Number(label="PyTorch CNN epochs", value=5, precision=0, minimum=1)
                sync_wait = gr.Number(label="Post-write sync wait (seconds)", value=0, precision=0, minimum=0)
            train_btn = gr.Button("Train models in this session", variant="primary")
            train_status = gr.Textbox(label="Training status", lines=2)
            train_logs = gr.Textbox(label="Training logs", lines=20, max_lines=30)

        inputs = [source, pretrained_dir, session_dir, custom_dir]
        refresh.click(artifact_status, inputs=inputs, outputs=[artifact_md, model_name], queue=False)
        source.change(artifact_status, inputs=inputs, outputs=[artifact_md, model_name], queue=False)
        pretrained_dir.change(artifact_status, inputs=inputs, outputs=[artifact_md, model_name], queue=False)
        session_dir.change(artifact_status, inputs=inputs, outputs=[artifact_md, model_name], queue=False)
        custom_dir.change(artifact_status, inputs=inputs, outputs=[artifact_md, model_name], queue=False)

        random_btn.click(random_index, inputs=[data_dir], outputs=[index]).then(
            predict,
            inputs=[data_dir, source, pretrained_dir, session_dir, custom_dir, model_name, index],
            outputs=[image, prediction, probs],
        )
        predict_btn.click(
            predict,
            inputs=[data_dir, source, pretrained_dir, session_dir, custom_dir, model_name, index],
            outputs=[image, prediction, probs],
        )
        train_btn.click(
            run_training_with_progress,
            inputs=[data_dir, session_dir, numpy_epochs, torch_epochs, sync_wait],
            outputs=[train_status, train_logs],
        )
        # Avoid doing connector/model reads automatically on page load. Some
        # mounted connectors can be slow to respond, and an automatic queued
        # prediction makes the Gradio UI look like it is stuck on "running".
        # Users explicitly click Refresh/Predict instead.
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MNIST Gradio dashboard")
    parser.add_argument("--server_port", default=int(os.environ.get("PORT", "8080")), type=int)
    parser.add_argument("--server_name", default="0.0.0.0", type=str)
    parser.add_argument("--root_path", default=os.environ.get("RENKU_BASE_URL_PATH") or None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app().queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        root_path=args.root_path,
        show_api=False,
    )
