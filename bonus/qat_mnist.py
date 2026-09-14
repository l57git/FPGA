#!/usr/bin/env python3
"""Quantization-aware training for the LeNet MNIST model.

The experiment is intentionally independent from HLS: it answers whether fake
quantization during training can reduce the low-bit PTQ error before the team
decides whether to export another fixed-point hardware variant.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised by environment only
    raise SystemExit(
        "QAT experiment requires numpy and torch. On the course VM use "
        "/home/ubuntu/workspace/lenet_level1/.venv/bin/python."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
BATCH_MAGIC = -20260902
MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049

PARAMETER_SPECS: Tuple[Tuple[str, int, Tuple[int, ...]], ...] = (
    ("conv1_w", 150, (6, 5, 5)),
    ("conv1_b", 6, (6,)),
    ("conv2_w", 2400, (16, 6, 5, 5)),
    ("conv2_b", 16, (16,)),
    ("fc1_w", 30720, (120, 256)),
    ("fc1_b", 120, (120,)),
    ("fc2_w", 10080, (84, 120)),
    ("fc2_b", 84, (84,)),
    ("fc3_w", 840, (10, 84)),
    ("fc3_b", 10, (10,)),
)


@dataclass(frozen=True)
class FixedPointSpec:
    bits: int
    integer_bits: int
    signed: bool = True

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.integer_bits <= 0:
            raise ValueError("integer_bits must be positive")
        if self.integer_bits > self.bits:
            raise ValueError("integer_bits cannot exceed bits")
        if self.signed and self.bits < 2:
            raise ValueError("signed fixed point needs at least two bits")

    @property
    def fraction_bits(self) -> int:
        return self.bits - self.integer_bits

    @property
    def qmin(self) -> int:
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    @property
    def step(self) -> float:
        return 1.0 / float(1 << self.fraction_bits)

    @property
    def minimum(self) -> float:
        return self.qmin * self.step

    @property
    def maximum(self) -> float:
        return self.qmax * self.step

    def as_dict(self) -> Dict[str, object]:
        return {
            "bits": self.bits,
            "integer_bits": self.integer_bits,
            "fraction_bits": self.fraction_bits,
            "signed": self.signed,
            "step": self.step,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lenet_parameters(blob_path: Path) -> Dict[str, np.ndarray]:
    path = Path(blob_path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"blob is too short: {path}")
    magic, sample_count = struct.unpack_from("<ii", raw, 0)
    if magic != BATCH_MAGIC:
        raise ValueError(f"bad blob magic {magic}, expected {BATCH_MAGIC}")
    if sample_count <= 0:
        raise ValueError(f"bad sample count in blob: {sample_count}")

    offset = 8
    parameters: Dict[str, np.ndarray] = {}
    for name, count, shape in PARAMETER_SPECS:
        end = offset + 4 * count
        if end > len(raw):
            raise ValueError(f"blob ended while reading {name}: {path}")
        values = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy()
        parameters[name] = values.reshape(shape)
        offset = end
    return parameters


def _read_maybe_gzip(path: Path) -> bytes:
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    gzip_path = path.with_suffix(path.suffix + ".gz")
    if gzip_path.exists():
        with gzip.open(gzip_path, "rb") as stream:
            return stream.read()
    raise FileNotFoundError(f"missing MNIST file: {path} or {gzip_path}")


def find_mnist_raw_dir(data_dir: Path) -> Path:
    root = Path(data_dir)
    candidates = (
        root,
        root / "MNIST" / "raw",
        root / "raw",
    )
    for candidate in candidates:
        image = candidate / "train-images-idx3-ubyte"
        label = candidate / "train-labels-idx1-ubyte"
        if image.exists() or image.with_suffix(image.suffix + ".gz").exists():
            if label.exists() or label.with_suffix(label.suffix + ".gz").exists():
                return candidate
    raise FileNotFoundError(
        f"cannot find MNIST raw files under {root}; expected MNIST/raw/*.idx*-ubyte"
    )


def read_idx_images(path: Path, expected_count: int) -> np.ndarray:
    raw = _read_maybe_gzip(path)
    if len(raw) < 16:
        raise ValueError(f"MNIST image file is too short: {path}")
    magic, count, rows, columns = struct.unpack_from(">IIII", raw, 0)
    if magic != MNIST_IMAGE_MAGIC:
        raise ValueError(f"bad image magic {magic}, expected {MNIST_IMAGE_MAGIC}")
    if count != expected_count or (rows, columns) != (28, 28):
        raise ValueError(
            f"unexpected image header count={count}, shape={rows}x{columns}"
        )
    expected_size = 16 + count * rows * columns
    if len(raw) != expected_size:
        raise ValueError(f"bad image file length {len(raw)}, expected {expected_size}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(count, rows, columns)


def read_idx_labels(path: Path, expected_count: int) -> np.ndarray:
    raw = _read_maybe_gzip(path)
    if len(raw) < 8:
        raise ValueError(f"MNIST label file is too short: {path}")
    magic, count = struct.unpack_from(">II", raw, 0)
    if magic != MNIST_LABEL_MAGIC:
        raise ValueError(f"bad label magic {magic}, expected {MNIST_LABEL_MAGIC}")
    if count != expected_count:
        raise ValueError(f"unexpected label count {count}, expected {expected_count}")
    expected_size = 8 + count
    if len(raw) != expected_size:
        raise ValueError(f"bad label file length {len(raw)}, expected {expected_size}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be 0..9")
    return labels.astype(np.int64)


def make_dataset(raw_dir: Path, train: bool, limit: int | None) -> TensorDataset:
    prefix = "train" if train else "t10k"
    count = 60000 if train else 10000
    images = read_idx_images(raw_dir / f"{prefix}-images-idx3-ubyte", count)
    labels = read_idx_labels(raw_dir / f"{prefix}-labels-idx1-ubyte", count)
    if limit is not None:
        if limit <= 0 or limit > count:
            raise ValueError(f"bad MNIST limit {limit}; expected 1..{count}")
        images = images[:limit]
        labels = labels[:limit]
    image_tensor = torch.from_numpy(images).to(torch.float32).unsqueeze(1) / 255.0
    label_tensor = torch.from_numpy(labels).to(torch.long)
    return TensorDataset(image_tensor, label_tensor)


def fake_quantize(values: torch.Tensor, spec: FixedPointSpec) -> torch.Tensor:
    scale = float(1 << spec.fraction_bits)
    quantized = torch.floor(values * scale + 0.5).clamp(spec.qmin, spec.qmax) / scale
    return values + (quantized - values).detach()


def hard_quantize(values: torch.Tensor, spec: FixedPointSpec) -> torch.Tensor:
    scale = float(1 << spec.fraction_bits)
    return torch.floor(values * scale + 0.5).clamp(spec.qmin, spec.qmax) / scale


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QuantAwareLeNet(LeNet):
    def __init__(
        self,
        weight_spec: FixedPointSpec,
        activation_spec: FixedPointSpec,
        quantize_activations: bool = True,
        quantize_bias: bool = False,
    ) -> None:
        super().__init__()
        self.weight_spec = weight_spec
        self.activation_spec = activation_spec
        self.quantize_activations = quantize_activations
        self.quantize_bias = quantize_bias

    def _qw(self, values: torch.Tensor) -> torch.Tensor:
        return fake_quantize(values, self.weight_spec)

    def _qb(self, values: torch.Tensor | None) -> torch.Tensor | None:
        if values is None or not self.quantize_bias:
            return values
        return fake_quantize(values, self.activation_spec)

    def _qa(self, values: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations:
            return values
        return fake_quantize(values, self.activation_spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._qa(x)
        x = F.conv2d(x, self._qw(self.conv1.weight), self._qb(self.conv1.bias))
        x = self.pool(torch.relu(self._qa(x)))
        x = F.conv2d(x, self._qw(self.conv2.weight), self._qb(self.conv2.bias))
        x = self.pool(torch.relu(self._qa(x)))
        x = torch.flatten(x, 1)
        x = F.linear(x, self._qw(self.fc1.weight), self._qb(self.fc1.bias))
        x = torch.relu(self._qa(x))
        x = F.linear(x, self._qw(self.fc2.weight), self._qb(self.fc2.bias))
        x = torch.relu(self._qa(x))
        x = F.linear(x, self._qw(self.fc3.weight), self._qb(self.fc3.bias))
        return self._qa(x)


def load_parameters(model: LeNet, parameters: Mapping[str, np.ndarray]) -> None:
    with torch.no_grad():
        model.conv1.weight.copy_(torch.from_numpy(parameters["conv1_w"][:, None, :, :]))
        model.conv1.bias.copy_(torch.from_numpy(parameters["conv1_b"]))
        model.conv2.weight.copy_(torch.from_numpy(parameters["conv2_w"]))
        model.conv2.bias.copy_(torch.from_numpy(parameters["conv2_b"]))
        model.fc1.weight.copy_(torch.from_numpy(parameters["fc1_w"]))
        model.fc1.bias.copy_(torch.from_numpy(parameters["fc1_b"]))
        model.fc2.weight.copy_(torch.from_numpy(parameters["fc2_w"]))
        model.fc2.bias.copy_(torch.from_numpy(parameters["fc2_b"]))
        model.fc3.weight.copy_(torch.from_numpy(parameters["fc3_w"]))
        model.fc3.bias.copy_(torch.from_numpy(parameters["fc3_b"]))


def iter_weight_tensors(model: LeNet) -> Iterable[Tuple[str, torch.Tensor]]:
    for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        yield f"{name}.weight", layer.weight


def iter_bias_tensors(model: LeNet) -> Iterable[Tuple[str, torch.Tensor]]:
    for name in ("conv1", "conv2", "fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        yield f"{name}.bias", layer.bias


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    labels: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    for images, batch_labels in loader:
        images = images.to(device)
        output = model(images).detach().cpu()
        logits.append(output)
        labels.append(batch_labels.detach().cpu())
    all_labels = torch.cat(labels)
    all_logits = torch.cat(logits)
    predictions = all_logits.argmax(dim=1)
    return all_labels, predictions, all_logits


def logits_metrics(
    name: str,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    logits: torch.Tensor,
    float_predictions: torch.Tensor | None = None,
    float_logits: torch.Tensor | None = None,
) -> Dict[str, object]:
    sample_count = int(labels.numel())
    correct = int((predictions == labels).sum().item())
    row: Dict[str, object] = {
        "model": name,
        "sample_count": sample_count,
        "correct": correct,
        "accuracy_percent": 100.0 * correct / sample_count,
    }
    if float_predictions is not None:
        matches = int((predictions == float_predictions).sum().item())
        row["prediction_matches_float"] = matches
        row["prediction_match_float_percent"] = 100.0 * matches / sample_count
    if float_logits is not None:
        diff = logits.to(torch.float64) - float_logits.to(torch.float64)
        row["logit_mae_vs_float"] = float(diff.abs().mean().item())
        row["logit_mse_vs_float"] = float((diff * diff).mean().item())
        row["logit_rmse_vs_float"] = float(torch.sqrt((diff * diff).mean()).item())
        row["logit_max_abs_error_vs_float"] = float(diff.abs().max().item())
    return row


def weight_error_rows(
    stage: str,
    model: LeNet,
    weight_spec: FixedPointSpec,
    bias_spec: FixedPointSpec,
    quantize_bias: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_count = 0
    total_sq_error = 0.0

    tensors: List[Tuple[str, torch.Tensor, FixedPointSpec]] = [
        (name, tensor, weight_spec) for name, tensor in iter_weight_tensors(model)
    ]
    if quantize_bias:
        tensors += [(name, tensor, bias_spec) for name, tensor in iter_bias_tensors(model)]

    with torch.no_grad():
        for name, tensor, tensor_spec in tensors:
            original = tensor.detach().cpu().to(torch.float64)
            quantized = hard_quantize(original, tensor_spec)
            diff = quantized - original
            count = int(original.numel())
            mse = float((diff * diff).mean().item())
            max_abs_error = float(diff.abs().max().item())
            saturation_count = int(
                ((original < tensor_spec.minimum) | (original > tensor_spec.maximum))
                .sum()
                .item()
            )
            rows.append(
                {
                    "stage": stage,
                    "tensor": name,
                    "count": count,
                    "bits": tensor_spec.bits,
                    "integer_bits": tensor_spec.integer_bits,
                    "fraction_bits": tensor_spec.fraction_bits,
                    "step": tensor_spec.step,
                    "mse_to_latent": mse,
                    "max_abs_error_to_latent": max_abs_error,
                    "saturation_count": saturation_count,
                }
            )
            total_count += count
            total_sq_error += mse * count

    aggregate = {
        "stage": stage,
        "tensor": "ALL",
        "count": total_count,
        "bits": weight_spec.bits,
        "integer_bits": weight_spec.integer_bits,
        "fraction_bits": weight_spec.fraction_bits,
        "step": weight_spec.step,
        "mse_to_latent": total_sq_error / total_count,
        "max_abs_error_to_latent": max(
            float(row["max_abs_error_to_latent"]) for row in rows
        ),
        "saturation_count": sum(int(row["saturation_count"]) for row in rows),
    }
    return rows, aggregate


def quantized_state_dict(
    model: QuantAwareLeNet,
    weight_spec: FixedPointSpec,
    activation_spec: FixedPointSpec,
    quantize_bias: bool,
) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith(".weight"):
                state[name] = hard_quantize(tensor.cpu(), weight_spec)
            elif name.endswith(".bias") and quantize_bias:
                state[name] = hard_quantize(tensor.cpu(), activation_spec)
            else:
                state[name] = tensor.cpu().clone()
    return state


def train_qat(
    model: QuantAwareLeNet,
    teacher: LeNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    distill_weight: float,
    distill_temperature: float,
    max_train_batches: int | None,
) -> List[Dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    teacher.eval()

    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        loss_total = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_index > max_train_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            hard_loss = ce_loss(logits, labels)
            if distill_weight > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                temperature = distill_temperature
                student_log_probs = F.log_softmax(logits / temperature, dim=1)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
                soft_loss = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (1.0 - distill_weight) * hard_loss + distill_weight * soft_loss
            else:
                loss = hard_loss
            loss.backward()
            optimizer.step()

            predictions = logits.detach().argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            loss_total += float(loss.detach().item()) * int(labels.numel())

        labels_cpu, predictions_cpu, _ = collect_logits(model, test_loader, device)
        test_correct = int((predictions_cpu == labels_cpu).sum().item())
        history.append(
            {
                "epoch": epoch,
                "train_samples": total,
                "train_loss": loss_total / max(total, 1),
                "train_accuracy_percent": 100.0 * correct / max(total, 1),
                "test_accuracy_percent": 100.0 * test_correct / labels_cpu.numel(),
                "test_correct": test_correct,
                "test_samples": int(labels_cpu.numel()),
            }
        )
        print(
            f"epoch {epoch}: "
            f"train_acc={history[-1]['train_accuracy_percent']:.2f}% "
            f"test_acc={history[-1]['test_accuracy_percent']:.2f}%"
        )
    return history


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt_percent(value: object) -> str:
    return f"{float(value):.2f}%"


def _fmt_float(value: object) -> str:
    return f"{float(value):.6g}"


def write_report(
    path: Path,
    summary: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
    aggregate_weight_rows: Sequence[Mapping[str, object]],
) -> None:
    float_acc = float(metrics_rows[0]["accuracy_percent"])
    lines = [
        "# Bonus QAT量化误差实验",
        "",
        "## 方法",
        "",
        (
            "PTQ行直接把已有LeNet参数映射到同一组定点网格；QAT行从相同参数初始化，"
            "训练前向对权重和激活插入fake quant，反向使用STE直通梯度。"
        ),
        "",
        "## 配置",
        "",
        f"- 训练集：MNIST train，{summary['train_samples']}张",
        f"- 测试集：MNIST test，{summary['test_samples']}张",
        f"- 权重量化：{summary['weight_spec']}",
        f"- 激活/输出量化：{summary['activation_spec']}",
        f"- QAT epoch：{summary['epochs']}，batch size：{summary['batch_size']}",
        "",
        "## 结果",
        "",
        "|模型|准确率|相对浮点损失(pp)|预测与浮点一致率|logit MAE|logit RMSE|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        loss = float_acc - float(row["accuracy_percent"])
        match = row.get("prediction_match_float_percent", "")
        mae = row.get("logit_mae_vs_float", "")
        rmse = row.get("logit_rmse_vs_float", "")
        lines.append(
            "|{model}|{accuracy}|{loss:.2f}|{match}|{mae}|{rmse}|".format(
                model=row["model"],
                accuracy=_fmt_percent(row["accuracy_percent"]),
                loss=loss,
                match=_fmt_percent(match) if match != "" else "",
                mae=_fmt_float(mae) if mae != "" else "",
                rmse=_fmt_float(rmse) if rmse != "" else "",
            )
        )
    lines += [
        "",
        "## 权重量化误差",
        "",
        "|阶段|元素数|MSE(量化值-潜变量)|最大绝对误差|饱和数|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_weight_rows:
        lines.append(
            "|{stage}|{count}|{mse}|{maxerr}|{sat}|".format(
                stage=row["stage"],
                count=row["count"],
                mse=_fmt_float(row["mse_to_latent"]),
                maxerr=_fmt_float(row["max_abs_error_to_latent"]),
                sat=row["saturation_count"],
            )
        )
    lines += [
        "",
        "## 结论边界",
        "",
        (
            "本实验是软件QAT证据：说明低比特定点网格下经过训练适配后的误差变化。"
            "它尚未替换HLS源码中的权重数组，也未报告新的综合资源或时序。"
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory containing MNIST/raw or the raw IDX files",
    )
    parser.add_argument(
        "--init-blob",
        type=Path,
        default=Path("level1/data/lenet_accuracy_1.bin"),
        help="LeNet blob providing the initial float parameters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bonus/results/w8a8_i6_qat"),
    )
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--weight-integer-bits", type=int, default=6)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--activation-integer-bits", type=int, default=6)
    parser.add_argument(
        "--no-activation-quant",
        action="store_true",
        help="quantize weights only; activations/logits stay floating point",
    )
    parser.add_argument(
        "--quantize-bias",
        action="store_true",
        help="also fake-quantize biases; off by default to match mixed_precision",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="skip writing QAT checkpoint files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not 0.0 <= args.distill_weight < 1.0:
        raise ValueError("distill-weight must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    raw_dir = find_mnist_raw_dir(data_dir)
    init_blob = resolve_repo_path(args.init_blob)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_spec = FixedPointSpec(args.weight_bits, args.weight_integer_bits)
    activation_spec = FixedPointSpec(args.activation_bits, args.activation_integer_bits)

    train_set = make_dataset(raw_dir, train=True, limit=args.train_limit)
    test_set = make_dataset(raw_dir, train=False, limit=args.test_limit)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    parameters = read_lenet_parameters(init_blob)
    float_model = LeNet().to(device)
    ptq_model = QuantAwareLeNet(
        weight_spec,
        activation_spec,
        quantize_activations=not args.no_activation_quant,
        quantize_bias=args.quantize_bias,
    ).to(device)
    qat_model = QuantAwareLeNet(
        weight_spec,
        activation_spec,
        quantize_activations=not args.no_activation_quant,
        quantize_bias=args.quantize_bias,
    ).to(device)
    load_parameters(float_model, parameters)
    load_parameters(ptq_model, parameters)
    load_parameters(qat_model, parameters)

    labels, float_predictions, float_logits = collect_logits(float_model, test_loader, device)
    _, ptq_predictions, ptq_logits = collect_logits(ptq_model, test_loader, device)
    history = train_qat(
        qat_model,
        float_model,
        train_loader,
        test_loader,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.distill_weight,
        args.distill_temperature,
        args.max_train_batches,
    )
    _, qat_predictions, qat_logits = collect_logits(qat_model, test_loader, device)

    metrics_rows = [
        logits_metrics("float", labels, float_predictions, float_logits),
        logits_metrics(
            "ptq",
            labels,
            ptq_predictions,
            ptq_logits,
            float_predictions,
            float_logits,
        ),
        logits_metrics(
            "qat",
            labels,
            qat_predictions,
            qat_logits,
            float_predictions,
            float_logits,
        ),
    ]

    ptq_weight_rows, ptq_weight_aggregate = weight_error_rows(
        "ptq", ptq_model, weight_spec, activation_spec, args.quantize_bias
    )
    qat_weight_rows, qat_weight_aggregate = weight_error_rows(
        "qat", qat_model, weight_spec, activation_spec, args.quantize_bias
    )
    weight_rows = ptq_weight_rows + qat_weight_rows
    aggregate_weight_rows = [ptq_weight_aggregate, qat_weight_aggregate]

    summary = {
        "experiment": "LeNet MNIST QAT bonus",
        "device": str(device),
        "seed": args.seed,
        "mnist_raw_dir": str(raw_dir),
        "init_blob": str(init_blob),
        "init_blob_sha256": sha256_file(init_blob),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "distill_weight": args.distill_weight,
        "distill_temperature": args.distill_temperature,
        "quantize_activations": not args.no_activation_quant,
        "quantize_bias": args.quantize_bias,
        "weight_spec": weight_spec.as_dict(),
        "activation_spec": activation_spec.as_dict(),
        "history": history,
        "metrics": metrics_rows,
        "weight_quantization_error": aggregate_weight_rows,
    }

    if not args.no_save_model:
        torch.save(qat_model.cpu().state_dict(), output_dir / "qat_state.pt")
        torch.save(
            quantized_state_dict(qat_model, weight_spec, activation_spec, args.quantize_bias),
            output_dir / "qat_quantized_state.pt",
        )
        summary["qat_state"] = str(output_dir / "qat_state.pt")
        summary["qat_quantized_state"] = str(output_dir / "qat_quantized_state.pt")

    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "weight_quantization_error.csv", weight_rows)
    write_json(output_dir / "summary.json", summary)
    write_report(
        output_dir / "experiment_report.md",
        summary,
        metrics_rows,
        aggregate_weight_rows,
    )

    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
