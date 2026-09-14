#!/usr/bin/env python3
"""Export a QAT checkpoint and MNIST test split into the LeNet HLS blob format."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict

import torch

BONUS_DIR = Path(__file__).resolve().parent
if str(BONUS_DIR) not in sys.path:
    sys.path.insert(0, str(BONUS_DIR))

from qat_mnist import (  # noqa: E402
    BATCH_MAGIC,
    FixedPointSpec,
    LeNet,
    PARAMETER_SPECS,
    find_mnist_raw_dir,
    hard_quantize,
    make_dataset,
    sha256_file,
)


ROOT = BONUS_DIR.parent


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def model_parameters(
    state_path: Path,
    data_spec: FixedPointSpec,
    quantize_before_write: bool,
) -> Dict[str, torch.Tensor]:
    state = torch.load(state_path, map_location="cpu")
    model = LeNet()
    model.load_state_dict(state)
    model.eval()
    tensors: Dict[str, torch.Tensor] = {
        "conv1_w": model.conv1.weight.detach().cpu().squeeze(1),
        "conv1_b": model.conv1.bias.detach().cpu(),
        "conv2_w": model.conv2.weight.detach().cpu(),
        "conv2_b": model.conv2.bias.detach().cpu(),
        "fc1_w": model.fc1.weight.detach().cpu(),
        "fc1_b": model.fc1.bias.detach().cpu(),
        "fc2_w": model.fc2.weight.detach().cpu(),
        "fc2_b": model.fc2.bias.detach().cpu(),
        "fc3_w": model.fc3.weight.detach().cpu(),
        "fc3_b": model.fc3.bias.detach().cpu(),
    }
    if quantize_before_write:
        tensors = {name: hard_quantize(tensor, data_spec) for name, tensor in tensors.items()}
    return tensors


def write_blob(
    output: Path,
    parameters: Dict[str, torch.Tensor],
    dataset: torch.utils.data.TensorDataset,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(struct.pack("<ii", BATCH_MAGIC, len(dataset)))
        for name, count, shape in PARAMETER_SPECS:
            tensor = parameters[name].detach().cpu().contiguous()
            if tuple(tensor.shape) != shape or tensor.numel() != count:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
            stream.write(tensor.numpy().astype("<f4").reshape(-1).tobytes())
        for image, label in dataset:
            stream.write(struct.pack("<i", int(label)))
            stream.write(image.numpy().astype("<f4").reshape(-1).tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("bonus/results/w8a8_i6_qat/qat_quantized_state.pt"),
        help="QAT state_dict produced by qat_mnist.py",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory containing MNIST/raw or raw IDX files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bonus/results/w8a8_i6_qat_hls/qat_hls_input.bin"),
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--data-bits", type=int, default=8)
    parser.add_argument("--data-integer-bits", type=int, default=6)
    parser.add_argument(
        "--no-quantize-before-write",
        action="store_true",
        help="write latent checkpoint values and let the C++ testbench quantize on read",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_path = resolve_repo_path(args.state)
    output_path = resolve_repo_path(args.output)
    metadata_path = resolve_repo_path(args.metadata) if args.metadata else output_path.with_suffix(".json")
    data_spec = FixedPointSpec(args.data_bits, args.data_integer_bits)
    raw_dir = find_mnist_raw_dir(args.data_dir)
    dataset = make_dataset(raw_dir, train=False, limit=args.test_limit)
    parameters = model_parameters(
        state_path,
        data_spec,
        quantize_before_write=not args.no_quantize_before_write,
    )
    write_blob(output_path, parameters, dataset)
    metadata = {
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "sample_count": len(dataset),
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "mnist_raw_dir": str(raw_dir),
        "data_spec": data_spec.as_dict(),
        "quantize_before_write": not args.no_quantize_before_write,
        "protocol_magic": BATCH_MAGIC,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
