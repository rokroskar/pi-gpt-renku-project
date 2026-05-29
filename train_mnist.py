#!/usr/bin/env python3
"""Train an MNIST classifier for a Renku non-interactive job demo.

The script expects the MNIST IDX files from Zenodo DOI 10.5281/zenodo.10058130.
Inside Renku, use the DOI data connector mount, e.g.
`/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130`.
For local testing only, set --download-if-missing to fetch the same files from Zenodo.
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


def _limit_or_none(limit: int | None) -> int | None:
    return None if limit is None or limit <= 0 else limit


def read_images(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        n = min(count, limit) if _limit_or_none(limit) else count
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_labels(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        n = min(count, limit) if _limit_or_none(limit) else count
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    h1_pre = x @ params["w1"] + params["b1"]
    h1 = np.maximum(h1_pre, 0.0)
    h2_pre = h1 @ params["w2"] + params["b2"]
    h2 = np.maximum(h2_pre, 0.0)
    logits = h2 @ params["w3"] + params["b3"]
    return logits, (h1, h2)


def accuracy(x: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], batch_size: int = 2048) -> float:
    correct = 0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits, _ = forward(xb, params)
        correct += int(np.sum(logits.argmax(axis=1) == yb))
    return correct / len(x)


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    m: dict[str, np.ndarray],
    v: dict[str, np.ndarray],
    step: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    for name, param in params.items():
        m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
        v[name] = beta2 * v[name] + (1.0 - beta2) * (grads[name] * grads[name])
        m_hat = m[name] / (1.0 - beta1**step)
        v_hat = v[name] / (1.0 - beta2**step)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    if args.download_if_missing:
        download_missing(data_dir)

    x_train = read_images(data_dir / "train-images-idx3-ubyte", args.train_limit)
    y_train = read_labels(data_dir / "train-labels-idx1-ubyte", args.train_limit)
    x_test = read_images(data_dir / "t10k-images-idx3-ubyte", args.test_limit)
    y_test = read_labels(data_dir / "t10k-labels-idx1-ubyte", args.test_limit)

    rng = np.random.default_rng(args.seed)
    input_dim = x_train.shape[1]
    params = {
        "w1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, args.hidden1)).astype(np.float32),
        "b1": np.zeros(args.hidden1, dtype=np.float32),
        "w2": rng.normal(0.0, np.sqrt(2.0 / args.hidden1), size=(args.hidden1, args.hidden2)).astype(np.float32),
        "b2": np.zeros(args.hidden2, dtype=np.float32),
        "w3": rng.normal(0.0, np.sqrt(2.0 / args.hidden2), size=(args.hidden2, 10)).astype(np.float32),
        "b3": np.zeros(10, dtype=np.float32),
    }
    m = {name: np.zeros_like(value) for name, value in params.items()}
    v = {name: np.zeros_like(value) for name, value in params.items()}

    print(
        f"Loaded MNIST: train={x_train.shape}, test={x_test.shape}, "
        f"model=mlp({args.hidden1},{args.hidden2}), epochs={args.epochs}, batch_size={args.batch_size}",
        flush=True,
    )

    start_time = time.time()
    step = 0
    best_test_accuracy = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        lr = args.learning_rate * (args.lr_decay ** (epoch - 1))
        order = rng.permutation(len(x_train))
        losses = []
        for start_idx in range(0, len(x_train), args.batch_size):
            step += 1
            batch_idx = order[start_idx : start_idx + args.batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            batch_n = len(yb)

            logits, (h1, h2) = forward(xb, params)
            probs = softmax(logits)
            ce_loss = -np.log(probs[np.arange(batch_n), yb] + 1e-12).mean()
            l2_loss = 0.5 * args.l2 * (
                np.sum(params["w1"] * params["w1"])
                + np.sum(params["w2"] * params["w2"])
                + np.sum(params["w3"] * params["w3"])
            )
            losses.append(float(ce_loss + l2_loss))

            probs[np.arange(batch_n), yb] -= 1.0
            probs /= batch_n
            grads: dict[str, np.ndarray] = {}
            grads["w3"] = h2.T @ probs + args.l2 * params["w3"]
            grads["b3"] = probs.sum(axis=0)
            dh2 = probs @ params["w3"].T
            dh2[h2 <= 0.0] = 0.0
            grads["w2"] = h1.T @ dh2 + args.l2 * params["w2"]
            grads["b2"] = dh2.sum(axis=0)
            dh1 = dh2 @ params["w2"].T
            dh1[h1 <= 0.0] = 0.0
            grads["w1"] = xb.T @ dh1 + args.l2 * params["w1"]
            grads["b1"] = dh1.sum(axis=0)

            adam_update(params, grads, m, v, step, lr)

        train_acc = accuracy(x_train, y_train, params)
        test_acc = accuracy(x_test, y_test, params)
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_epoch = epoch
        print(
            f"epoch={epoch:02d} lr={lr:.6f} loss={np.mean(losses):.4f} "
            f"train_accuracy={train_acc:.4f} test_accuracy={test_acc:.4f} "
            f"best_test_accuracy={best_test_accuracy:.4f}",
            flush=True,
        )

    metrics = {
        "model": f"mlp({args.hidden1},{args.hidden2})",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "train_accuracy": accuracy(x_train, y_train, params),
        "test_accuracy": accuracy(x_test, y_test, params),
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
        "duration_seconds": round(time.time() - start_time, 3),
        "data_dir": str(data_dir),
    }
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    model_path = model_dir / "mnist-mlp-model.npz"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(model_path, **params)
    np.savez_compressed(output_dir / "mnist-mlp-model.npz", **params)
    print(f"Wrote metrics to {metrics_path}", flush=True)
    print(f"Wrote model to {model_path}", flush=True)
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP classifier on MNIST")
    parser.add_argument("--data-dir", default=os.environ.get("MNIST_DATA_DIR", "data"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs"))
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "30")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.001")))
    parser.add_argument("--lr-decay", type=float, default=float(os.environ.get("LR_DECAY", "0.96")))
    parser.add_argument("--l2", type=float, default=float(os.environ.get("L2", "0.00005")))
    parser.add_argument("--hidden1", type=int, default=int(os.environ.get("HIDDEN1", "512")))
    parser.add_argument("--hidden2", type=int, default=int(os.environ.get("HIDDEN2", "256")))
    parser.add_argument("--train-limit", type=int, default=int(os.environ.get("TRAIN_LIMIT", "0")), help="0 means full train set")
    parser.add_argument("--test-limit", type=int, default=int(os.environ.get("TEST_LIMIT", "0")), help="0 means full test set")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--download-if-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
