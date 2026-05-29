#!/usr/bin/env python3
"""Train MNIST models and flush artifacts to a writable Renku data connector.

This script is intended for Renku non-interactive jobs. It writes both model
artifacts to a mounted output connector and then waits before exiting so the
rclone-backed connector has time to sync data to the remote backend.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST model artifacts for the Streamlit dashboard")
    parser.add_argument(
        "--data-dir",
        default="/home/renku/work/mnist-dataset-doi-10.5281-zenodo.10058130",
        help="Mounted MNIST data connector path",
    )
    parser.add_argument(
        "--artifact-dir",
        default="/home/renku/work/model-artifacts/mnist-models",
        help="Writable output connector path for model artifacts",
    )
    parser.add_argument("--numpy-epochs", type=int, default=30)
    parser.add_argument("--torch-epochs", type=int, default=5)
    parser.add_argument("--sync-wait", type=int, default=180, help="Seconds to wait after os.sync() for rclone flush")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "train_mnist.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        "outputs/numpy-mlp",
        "--model-dir",
        str(artifact_dir),
        "--epochs",
        str(args.numpy_epochs),
    ])
    run([
        sys.executable,
        "train_torch_mnist.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        "outputs/pytorch-cnn",
        "--model-dir",
        str(artifact_dir),
        "--epochs",
        str(args.torch_epochs),
    ])

    expected = [artifact_dir / "mnist-mlp-model.npz", artifact_dir / "mnist-small-cnn.pt"]
    manifest = {
        "artifact_dir": str(artifact_dir),
        "artifacts": [],
        "created_at_unix": time.time(),
    }
    for path in expected:
        size = path.stat().st_size
        with path.open("rb") as f:
            first_bytes = f.read(16).hex()
        print(f"Verified artifact {path} size={size} first16={first_bytes}", flush=True)
        manifest["artifacts"].append({"path": str(path), "size": size, "first16_hex": first_bytes})

    marker = artifact_dir / "_ARTIFACTS_READY.json"
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote readiness marker {marker}", flush=True)

    # Flush kernel page cache so rclone can observe all writes, then wait so the
    # asynchronous rclone mount can finish uploading to the remote connector.
    os.sync()
    print(f"Outputs written, waiting {args.sync_wait}s for rclone sync...", flush=True)
    time.sleep(args.sync_wait)

    print("Final artifact listing:", flush=True)
    for path in sorted(artifact_dir.iterdir()):
        print(f"  {path.name}\t{path.stat().st_size} bytes", flush=True)
    print("Artifact preparation complete.", flush=True)


if __name__ == "__main__":
    main()
