#!/usr/bin/env python3
"""Train a small PyTorch CNN on MNIST for a Renku non-interactive job demo.

Uses the MNIST IDX files from Zenodo DOI 10.5281/zenodo.10058130. In Renku,
pass the mounted DOI connector path, e.g.:

  /home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130

No data is downloaded unless --download-if-missing is explicitly provided.
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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ZENODO_BASE = "https://zenodo.org/records/10058130/files"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_missing(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
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


def effective_limit(limit: int | None) -> int | None:
    return None if limit is None or limit <= 0 else limit


def read_images(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        n = min(count, limit) if effective_limit(limit) else count
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0


def read_labels(path: Path, limit: int | None = None) -> np.ndarray:
    with open_idx(path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        n = min(count, limit) if effective_limit(limit) else count
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return loss_sum / total, correct / total


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if args.download_if_missing:
        download_missing(data_dir)

    x_train = read_images(data_dir / "train-images-idx3-ubyte", args.train_limit)
    y_train = read_labels(data_dir / "train-labels-idx1-ubyte", args.train_limit)
    x_test = read_images(data_dir / "t10k-images-idx3-ubyte", args.test_limit)
    y_test = read_labels(data_dir / "t10k-labels-idx1-ubyte", args.test_limit)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    eval_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    print(
        f"Loaded MNIST: train={tuple(x_train.shape)}, test={tuple(x_test.shape)}, "
        f"model=SmallCNN, epochs={args.epochs}, batch_size={args.batch_size}, device={device}",
        flush=True,
    )

    start_time = time.time()
    best_test_accuracy = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        scheduler.step()

        train_loss = loss_sum / total
        _, train_acc = evaluate(model, train_eval_loader, device)
        test_loss, test_acc = evaluate(model, eval_loader, device)
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_epoch = epoch
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:02d} lr={lr:.6f} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} train_accuracy={train_acc:.4f} "
            f"test_accuracy={test_acc:.4f} best_test_accuracy={best_test_accuracy:.4f}",
            flush=True,
        )

    final_test_loss, final_test_acc = evaluate(model, eval_loader, device)
    _, final_train_acc = evaluate(model, train_eval_loader, device)
    metrics = {
        "framework": "pytorch",
        "model": "SmallCNN",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_examples": int(len(train_ds)),
        "test_examples": int(len(test_ds)),
        "train_accuracy": final_train_acc,
        "test_accuracy": final_test_acc,
        "test_loss": final_test_loss,
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
        "duration_seconds": round(time.time() - start_time, 3),
        "device": str(device),
        "data_dir": str(data_dir),
    }

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    model_path = model_dir / "mnist-small-cnn.pt"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, model_path)
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, output_dir / "mnist-small-cnn.pt")
    print(f"Wrote metrics to {metrics_path}", flush=True)
    print(f"Wrote model to {model_path}", flush=True)
    print("FINAL_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN classifier on MNIST")
    parser.add_argument("--data-dir", default=os.environ.get("MNIST_DATA_DIR", "data"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs"))
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "5")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    parser.add_argument("--eval-batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "1024")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.001")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.0001")))
    parser.add_argument("--train-limit", type=int, default=int(os.environ.get("TRAIN_LIMIT", "0")), help="0 means full train set")
    parser.add_argument("--test-limit", type=int, default=int(os.environ.get("TEST_LIMIT", "0")), help="0 means full test set")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--download-if-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
