#!/usr/bin/env python3
"""Train a tiny MNIST classifier for a Renku non-interactive job demo.

The script expects the MNIST IDX files from Zenodo DOI 10.5281/zenodo.10058130.
When run inside Renku, the Zenodo data connector should mount these files under
`data/`. For local testing, set --download-if-missing to fetch them from Zenodo.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

ZENODO_BASE = "https://zenodo.org/records/10058130/files"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_missing(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = data_dir / filename
        if target.exists():
            continue
        url = f"{ZENODO_BASE}/{filename}?download=1"
        print(f"Downloading {url} -> {target}", flush=True)
        urllib.request.urlretrieve(url, target)


def open_idx(path: Path):
    if path.exists():
        return path.open("rb")
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rb")
    raise FileNotFoundError(f"Could not find {path} or {gz_path}")


def read_images(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        n = min(count, limit) if limit else count
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_labels(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        n = min(count, limit) if limit else count
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy(x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    pred = (x @ weights + bias).argmax(axis=1)
    return float(np.mean(pred == y))


def train(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    if args.download_if_missing:
        download_missing(data_dir)

    x_train = read_images(data_dir / "train-images-idx3-ubyte", args.train_limit)
    y_train = read_labels(data_dir / "train-labels-idx1-ubyte", args.train_limit)
    x_test = read_images(data_dir / "t10k-images-idx3-ubyte", args.test_limit)
    y_test = read_labels(data_dir / "t10k-labels-idx1-ubyte", args.test_limit)

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.01, size=(x_train.shape[1], 10)).astype(np.float32)
    bias = np.zeros(10, dtype=np.float32)

    print(
        f"Loaded MNIST: train={x_train.shape}, test={x_test.shape}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}",
        flush=True,
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        losses = []
        for start_idx in range(0, len(x_train), args.batch_size):
            batch_idx = order[start_idx : start_idx + args.batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            probs = softmax(xb @ weights + bias)
            losses.append(float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()))
            probs[np.arange(len(yb)), yb] -= 1.0
            weights -= args.learning_rate * (xb.T @ probs) / len(yb)
            bias -= args.learning_rate * probs.mean(axis=0)

        train_acc = accuracy(x_train, y_train, weights, bias)
        test_acc = accuracy(x_test, y_test, weights, bias)
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.4f} "
            f"train_accuracy={train_acc:.4f} test_accuracy={test_acc:.4f}",
            flush=True,
        )

    metrics = {
        "epochs": args.epochs,
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "train_accuracy": accuracy(x_train, y_train, weights, bias),
        "test_accuracy": accuracy(x_test, y_test, weights, bias),
        "duration_seconds": round(time.time() - start, 3),
        "data_dir": str(data_dir),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output_dir / "mnist-softmax-model.npz", weights=weights, bias=bias)
    print(f"Wrote metrics to {metrics_path}", flush=True)
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny softmax classifier on MNIST")
    parser.add_argument("--data-dir", default=os.environ.get("MNIST_DATA_DIR", "data"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "5")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.5")))
    parser.add_argument("--train-limit", type=int, default=int(os.environ.get("TRAIN_LIMIT", "12000")))
    parser.add_argument("--test-limit", type=int, default=int(os.environ.get("TEST_LIMIT", "2000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--download-if-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
