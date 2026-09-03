#!/usr/bin/env python3
"""Prepare and compare reproducible MNIST results for the Level 1 LeNet design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

BATCH_MAGIC = -20260902
MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049
EXPECTED_MNIST_COUNT = 10000
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
NORMALIZATION = "uint8 / 255.0"

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

RESULT_COLUMNS = ["index", "expected", "prediction"] + [
    f"logit_{index}" for index in range(10)
]


class ValidationError(ValueError):
    """Raised when an experiment input does not satisfy its file contract."""


@dataclass
class LeNetBlob:
    parameters: Dict[str, np.ndarray]
    labels: np.ndarray
    images: np.ndarray
    source_path: Path | None = None

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])


@dataclass
class MnistTestSet:
    images: np.ndarray
    labels: np.ndarray
    images_path: Path | None = None
    labels_path: Path | None = None

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parameters(parameters: Dict[str, np.ndarray]) -> None:
    expected_names = {name for name, _, _ in PARAMETER_SPECS}
    if set(parameters) != expected_names:
        raise ValidationError(
            "parameter names do not match the LeNet protocol: "
            f"expected {sorted(expected_names)}, got {sorted(parameters)}"
        )
    for name, count, shape in PARAMETER_SPECS:
        array = np.asarray(parameters[name])
        if array.size != count or tuple(array.shape) != shape:
            raise ValidationError(
                f"parameter {name} has shape {array.shape}, expected {shape}"
            )
        if not np.isfinite(array).all():
            raise ValidationError(f"parameter {name} contains non-finite values")


def _validate_samples(labels: np.ndarray, images: np.ndarray) -> None:
    labels = np.asarray(labels)
    images = np.asarray(images)
    if labels.ndim != 1:
        raise ValidationError(f"labels must be one-dimensional, got {labels.shape}")
    if images.shape != (labels.size, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValidationError(
            f"images have shape {images.shape}, expected "
            f"({labels.size}, {IMAGE_HEIGHT}, {IMAGE_WIDTH})"
        )
    if labels.size == 0:
        raise ValidationError("sample count must be positive")
    if np.any(labels < 0) or np.any(labels > 9):
        raise ValidationError("labels must be in the range 0..9")
    if not np.isfinite(images).all():
        raise ValidationError("images contain non-finite values")


def read_lenet_blob(path: Path, expected_count: int | None = None) -> LeNetBlob:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValidationError(f"blob is too short for header: {path}")

    magic, batch_count = struct.unpack_from("<ii", raw, 0)
    if magic != BATCH_MAGIC:
        raise ValidationError(
            f"invalid blob magic {magic}; expected {BATCH_MAGIC}"
        )
    if batch_count <= 0:
        raise ValidationError(f"batch count must be positive, got {batch_count}")
    if expected_count is not None and batch_count != expected_count:
        raise ValidationError(
            f"blob sample count is {batch_count}, expected {expected_count}"
        )

    parameter_bytes = sum(count for _, count, _ in PARAMETER_SPECS) * 4
    sample_bytes = 4 + IMAGE_PIXELS * 4
    expected_size = 8 + parameter_bytes + batch_count * sample_bytes
    if len(raw) != expected_size:
        raise ValidationError(
            f"invalid blob length {len(raw)}; expected {expected_size} "
            f"for {batch_count} samples"
        )

    parameters: Dict[str, np.ndarray] = {}
    offset = 8
    for name, count, shape in PARAMETER_SPECS:
        values = np.frombuffer(
            raw, dtype="<f4", count=count, offset=offset
        ).copy()
        parameters[name] = values.reshape(shape)
        offset += count * 4
    _validate_parameters(parameters)

    sample_dtype = np.dtype(
        [("label", "<i4"), ("image", "<f4", (IMAGE_PIXELS,))]
    )
    samples = np.frombuffer(
        raw, dtype=sample_dtype, count=batch_count, offset=offset
    )
    labels = samples["label"].copy()
    images = samples["image"].copy().reshape(
        batch_count, IMAGE_HEIGHT, IMAGE_WIDTH
    )
    _validate_samples(labels, images)

    return LeNetBlob(parameters, labels, images, path)


def write_lenet_blob(
    path: Path,
    parameters: Dict[str, np.ndarray],
    labels: np.ndarray,
    images: np.ndarray,
) -> None:
    _validate_parameters(parameters)
    labels = np.asarray(labels, dtype=np.int32)
    images = np.asarray(images, dtype=np.float32)
    _validate_samples(labels, images)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", BATCH_MAGIC, labels.size))
        for name, _, _ in PARAMETER_SPECS:
            stream.write(np.asarray(parameters[name], dtype="<f4").tobytes())
        sample_dtype = np.dtype(
            [("label", "<i4"), ("image", "<f4", (IMAGE_PIXELS,))]
        )
        samples = np.empty(labels.size, dtype=sample_dtype)
        samples["label"] = labels
        samples["image"] = images.reshape(labels.size, IMAGE_PIXELS)
        stream.write(samples.tobytes())


def read_idx_images(path: Path, expected_count: int = EXPECTED_MNIST_COUNT) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValidationError(f"MNIST image file is too short: {path}")
    magic, count, rows, columns = struct.unpack_from(">IIII", raw, 0)
    if magic != MNIST_IMAGE_MAGIC:
        raise ValidationError(
            f"invalid MNIST image magic {magic}; expected {MNIST_IMAGE_MAGIC}"
        )
    if count != expected_count:
        raise ValidationError(
            f"MNIST image count is {count}, expected {expected_count}"
        )
    if (rows, columns) != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValidationError(
            f"MNIST image dimensions are {rows}x{columns}, expected 28x28"
        )
    expected_size = 16 + count * rows * columns
    if len(raw) != expected_size:
        raise ValidationError(
            f"MNIST image file length is {len(raw)}, expected {expected_size}"
        )
    return np.frombuffer(raw, dtype=np.uint8, count=count * rows * columns, offset=16).copy().reshape(
        count, rows, columns
    )


def read_idx_labels(path: Path, expected_count: int = EXPECTED_MNIST_COUNT) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValidationError(f"MNIST label file is too short: {path}")
    magic, count = struct.unpack_from(">II", raw, 0)
    if magic != MNIST_LABEL_MAGIC:
        raise ValidationError(
            f"invalid MNIST label magic {magic}; expected {MNIST_LABEL_MAGIC}"
        )
    if count != expected_count:
        raise ValidationError(
            f"MNIST label count is {count}, expected {expected_count}"
        )
    expected_size = 8 + count
    if len(raw) != expected_size:
        raise ValidationError(
            f"MNIST label file length is {len(raw)}, expected {expected_size}"
        )
    labels = np.frombuffer(raw, dtype=np.uint8, count=count, offset=8).copy()
    if np.any(labels > 9):
        raise ValidationError("MNIST labels must be in the range 0..9")
    return labels.astype(np.int32)


def read_mnist_test_set(
    images_path: Path,
    labels_path: Path,
    expected_count: int = EXPECTED_MNIST_COUNT,
) -> MnistTestSet:
    images = read_idx_images(images_path, expected_count)
    labels = read_idx_labels(labels_path, expected_count)
    if images.shape[0] != labels.shape[0]:
        raise ValidationError(
            f"MNIST image/label counts differ: {images.shape[0]} vs {labels.shape[0]}"
        )
    return MnistTestSet(images, labels, Path(images_path), Path(labels_path))


def normalize_mnist_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise ValidationError(f"MNIST images must be uint8 before normalization, got {images.dtype}")
    return images.astype(np.float32) / np.float32(255.0)


def conv2d_valid(
    inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    if inputs.ndim != 4 or weights.ndim != 4:
        raise ValidationError("convolution inputs and weights must be four-dimensional")
    if inputs.shape[1] != weights.shape[1]:
        raise ValidationError(
            f"convolution channel mismatch: {inputs.shape[1]} vs {weights.shape[1]}"
        )
    if weights.shape[2] > inputs.shape[2] or weights.shape[3] > inputs.shape[3]:
        raise ValidationError("convolution kernel is larger than its input")
    if biases.shape != (weights.shape[0],):
        raise ValidationError(
            f"convolution bias shape is {biases.shape}, expected {(weights.shape[0],)}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(
        inputs, (weights.shape[2], weights.shape[3]), axis=(-2, -1)
    )
    output = np.einsum(
        "ncyxkl,ockl->noyx", windows, weights, optimize=True
    ).astype(np.float32, copy=False)
    output += biases[None, :, None, None]
    return output


def max_pool_2x2(inputs: np.ndarray) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 4 or inputs.shape[2] % 2 or inputs.shape[3] % 2:
        raise ValidationError("pool input must be NCHW with even spatial dimensions")
    rows = inputs.shape[2] // 2
    columns = inputs.shape[3] // 2
    return inputs.reshape(inputs.shape[0], inputs.shape[1], rows, 2, columns, 2).max(
        axis=(3, 5)
    )


def lenet_forward(
    images: np.ndarray, parameters: Dict[str, np.ndarray]
) -> np.ndarray:
    _validate_parameters(parameters)
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        if images.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValidationError(f"images have shape {images.shape}, expected N x 28 x 28")
        inputs = images[:, None, :, :]
    elif images.ndim == 4:
        if images.shape[1:] != (1, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValidationError(
                f"images have shape {images.shape}, expected N x 1 x 28 x 28"
            )
        inputs = images
    else:
        raise ValidationError("images must have shape N x 28 x 28 or N x 1 x 28 x 28")

    conv1 = conv2d_valid(
        inputs,
        parameters["conv1_w"][:, None, :, :],
        parameters["conv1_b"],
    )
    pool1 = max_pool_2x2(np.maximum(conv1, np.float32(0.0)))
    conv2 = conv2d_valid(
        pool1, parameters["conv2_w"], parameters["conv2_b"]
    )
    pool2 = max_pool_2x2(np.maximum(conv2, np.float32(0.0)))
    flattened = pool2.reshape(pool2.shape[0], -1)
    if flattened.shape[1] != 256:
        raise ValidationError(f"FC input has {flattened.shape[1]} values, expected 256")
    fc1 = np.maximum(
        flattened @ parameters["fc1_w"].T + parameters["fc1_b"],
        np.float32(0.0),
    )
    fc2 = np.maximum(
        fc1 @ parameters["fc2_w"].T + parameters["fc2_b"],
        np.float32(0.0),
    )
    return (
        fc2 @ parameters["fc3_w"].T + parameters["fc3_b"]
    ).astype(np.float32, copy=False)


def predict_images(
    images: np.ndarray,
    parameters: Dict[str, np.ndarray],
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValidationError("batch size must be positive")
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValidationError(f"images have shape {images.shape}, expected N x 28 x 28")
    logits = np.empty((images.shape[0], 10), dtype=np.float32)
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        logits[start:end] = lenet_forward(images[start:end], parameters)
    return logits, logits.argmax(axis=1).astype(np.int32)


def _write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_result_csv(
    path: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
) -> None:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    logits = np.asarray(logits)
    if labels.ndim != 1 or predictions.shape != labels.shape or logits.shape != (labels.size, 10):
        raise ValidationError("result arrays have incompatible shapes")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(RESULT_COLUMNS)
        for index in range(labels.size):
            writer.writerow(
                [index, int(labels[index]), int(predictions[index])]
                + [f"{float(value):.9g}" for value in logits[index]]
            )


def read_result_csv(path: Path) -> List[Dict[str, object]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValidationError(
                f"result header in {path} is {reader.fieldnames}, expected {RESULT_COLUMNS}"
            )
        rows: List[Dict[str, object]] = []
        for position, row in enumerate(reader):
            try:
                index = int(row["index"])
                expected = int(row["expected"])
                prediction = int(row["prediction"])
                logits = [float(row[f"logit_{i}"]) for i in range(10)]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValidationError(
                    f"invalid result row {position + 2} in {path}"
                ) from exc
            if index != position:
                raise ValidationError(
                    f"result index {index} at row {position} in {path}; expected {position}"
                )
            if not 0 <= expected <= 9 or not 0 <= prediction <= 9:
                raise ValidationError(f"invalid class in result row {position} in {path}")
            if not np.isfinite(logits).all():
                raise ValidationError(f"non-finite logits in result row {position} in {path}")
            rows.append(
                {
                    "index": index,
                    "expected": expected,
                    "prediction": prediction,
                    "logits": logits,
                }
            )
    if not rows:
        raise ValidationError(f"result file has no samples: {path}")
    return rows


def _blob_summary(blob: LeNetBlob) -> Dict[str, object]:
    result: Dict[str, object] = {
        "sample_count": blob.sample_count,
        "parameter_count": sum(count for _, count, _ in PARAMETER_SPECS),
        "parameter_names": [name for name, _, _ in PARAMETER_SPECS],
        "image_shape": list(blob.images.shape),
        "image_min": float(blob.images.min()),
        "image_max": float(blob.images.max()),
    }
    if blob.source_path is not None:
        result["source_path"] = str(blob.source_path)
        result["source_sha256"] = sha256_file(blob.source_path)
    return result


def command_inspect_blob(args: argparse.Namespace) -> int:
    blob = read_lenet_blob(Path(args.blob), args.expected_count)
    summary = _blob_summary(blob)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output:
        _write_json(Path(args.json_output), summary)
    return 0


def command_validate_mnist(args: argparse.Namespace) -> int:
    dataset = read_mnist_test_set(
        Path(args.images), Path(args.labels), args.expected_count
    )
    summary = {
        "sample_count": dataset.sample_count,
        "image_shape": list(dataset.images.shape),
        "label_min": int(dataset.labels.min()),
        "label_max": int(dataset.labels.max()),
        "image_min": int(dataset.images.min()),
        "image_max": int(dataset.images.max()),
        "normalization": NORMALIZATION,
        "images_path": str(dataset.images_path),
        "labels_path": str(dataset.labels_path),
        "images_sha256": sha256_file(Path(args.images)),
        "labels_sha256": sha256_file(Path(args.labels)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output:
        _write_json(Path(args.json_output), summary)
    return 0


def command_make_blob(args: argparse.Namespace) -> int:
    parameter_blob_path = Path(args.parameters)
    source_blob = read_lenet_blob(parameter_blob_path)
    dataset = read_mnist_test_set(
        Path(args.images), Path(args.labels), args.expected_count
    )
    normalized_images = normalize_mnist_images(dataset.images)
    output_path = Path(args.output)
    write_lenet_blob(
        output_path, source_blob.parameters, dataset.labels, normalized_images
    )
    summary = {
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "sample_count": dataset.sample_count,
        "parameter_count": sum(count for _, count, _ in PARAMETER_SPECS),
        "parameter_source": str(parameter_blob_path),
        "parameter_source_sha256": sha256_file(parameter_blob_path),
        "images_path": str(args.images),
        "images_sha256": sha256_file(Path(args.images)),
        "labels_path": str(args.labels),
        "labels_sha256": sha256_file(Path(args.labels)),
        "normalization": NORMALIZATION,
        "protocol_magic": BATCH_MAGIC,
    }
    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".metadata.json")
    _write_json(metadata_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_run_float(args: argparse.Namespace) -> int:
    blob = read_lenet_blob(Path(args.blob), args.expected_count)
    logits, predictions = predict_images(blob.images, blob.parameters, args.batch_size)
    output_path = Path(args.results)
    write_result_csv(output_path, blob.labels, predictions, logits)
    correct = int(np.count_nonzero(predictions == blob.labels))
    summary = {
        "result_path": str(output_path),
        "result_sha256": sha256_file(output_path),
        "source_blob": str(args.blob),
        "source_blob_sha256": sha256_file(Path(args.blob)),
        "sample_count": blob.sample_count,
        "correct": correct,
        "accuracy_percent": 100.0 * correct / blob.sample_count,
        "model": "NumPy float32 LeNet",
        "normalization": NORMALIZATION,
        "parameter_count": sum(count for _, count, _ in PARAMETER_SPECS),
    }
    if args.summary:
        _write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _write_mismatches(path: Path, mismatches: Iterable[Dict[str, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "index",
                "expected",
                "float_prediction",
                "hls_prediction",
                "float_correct",
                "hls_correct",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(mismatches)


def compare_result_files(
    float_path: Path,
    hls_path: Path,
    report_path: Path,
    mismatch_path: Path,
    threshold: float,
) -> Dict[str, object]:
    float_rows = read_result_csv(float_path)
    hls_rows = read_result_csv(hls_path)
    if len(float_rows) != len(hls_rows):
        raise ValidationError(
            f"result sample counts differ: float={len(float_rows)}, hls={len(hls_rows)}"
        )

    mismatches: List[Dict[str, int]] = []
    float_correct = 0
    hls_correct = 0
    prediction_matches = 0
    for float_row, hls_row in zip(float_rows, hls_rows):
        if float_row["index"] != hls_row["index"]:
            raise ValidationError(
                f"result index mismatch: float={float_row['index']}, hls={hls_row['index']}"
            )
        if float_row["expected"] != hls_row["expected"]:
            raise ValidationError(
                f"label mismatch at sample {float_row['index']}: "
                f"float={float_row['expected']}, hls={hls_row['expected']}"
            )
        expected = int(float_row["expected"])
        float_prediction = int(float_row["prediction"])
        hls_prediction = int(hls_row["prediction"])
        is_float_correct = float_prediction == expected
        is_hls_correct = hls_prediction == expected
        float_correct += int(is_float_correct)
        hls_correct += int(is_hls_correct)
        prediction_matches += int(float_prediction == hls_prediction)
        if float_prediction != hls_prediction:
            mismatches.append(
                {
                    "index": int(float_row["index"]),
                    "expected": expected,
                    "float_prediction": float_prediction,
                    "hls_prediction": hls_prediction,
                    "float_correct": int(is_float_correct),
                    "hls_correct": int(is_hls_correct),
                }
            )

    sample_count = len(float_rows)
    float_accuracy = 100.0 * float_correct / sample_count
    hls_accuracy = 100.0 * hls_correct / sample_count
    report = {
        "status": "PASS" if hls_accuracy >= threshold else "FAIL",
        "sample_count": sample_count,
        "float_correct": float_correct,
        "float_accuracy_percent": float_accuracy,
        "hls_correct": hls_correct,
        "hls_accuracy_percent": hls_accuracy,
        "absolute_accuracy_difference_percent": abs(float_accuracy - hls_accuracy),
        "prediction_matches": prediction_matches,
        "prediction_consistency_percent": 100.0 * prediction_matches / sample_count,
        "mismatch_count": len(mismatches),
        "hls_threshold_percent": threshold,
        "hls_threshold_passed": hls_accuracy >= threshold,
        "float_results": str(float_path),
        "hls_results": str(hls_path),
        "mismatches": str(mismatch_path),
    }
    _write_mismatches(mismatch_path, mismatches)
    _write_json(report_path, report)
    return report


def command_compare(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    mismatch_path = Path(args.mismatches) if args.mismatches else report_path.with_name(
        report_path.stem + "_mismatches.csv"
    )
    try:
        report = compare_result_files(
            Path(args.float_results),
            Path(args.hls_results),
            report_path,
            mismatch_path,
            args.threshold,
        )
    except (OSError, ValidationError) as exc:
        report = {
            "status": "ERROR",
            "error": str(exc),
            "float_results": str(args.float_results),
            "hls_results": str(args.hls_results),
            "hls_threshold_percent": args.threshold,
        }
        _write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["hls_threshold_passed"] else 1


def _read_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON metadata {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON metadata must contain an object: {path}")
    return value


def _summary_markdown(summary: Dict[str, object]) -> str:
    mnist = summary["mnist"]
    parameters = summary["parameters"]
    results = summary["results"]
    commands = summary["commands"]
    lines = [
        "# Level 1 MNIST 验证摘要",
        "",
        f"- 状态：**{summary['status']}**",
        f"- 样本数：{results['sample_count']}",
        f"- 归一化：`{summary['normalization']}`",
        f"- 定点数据类型：`{summary['fixed_point']['data_t']}`",
        f"- 定点累加类型：`{summary['fixed_point']['acc_t']}`",
        f"- HLS 工具：`{summary['hls_tool_version']}`",
        f"- 工具状态：{summary['toolchain_status']}",
        "",
        "## 数据摘要",
        "",
        f"- MNIST 图像：`{mnist['images_path']}`",
        f"- MNIST 图像 SHA-256：`{mnist['images_sha256']}`",
        f"- MNIST 标签：`{mnist['labels_path']}`",
        f"- MNIST 标签 SHA-256：`{mnist['labels_sha256']}`",
        f"- 参数源：`{parameters['source_path']}`",
        f"- 参数源 SHA-256：`{parameters['source_sha256']}`",
        f"- 参数数量：`{parameters['count']}`",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Python 浮点正确数 | {results['float_correct']} |",
        f"| Python 浮点准确率 | {results['float_accuracy_percent']:.2f}% |",
        f"| HLS 定点正确数 | {results['hls_correct']} |",
        f"| HLS 定点准确率 | {results['hls_accuracy_percent']:.2f}% |",
        f"| 准确率绝对差值 | {results['absolute_accuracy_difference_percent']:.2f}% |",
        f"| 预测一致率 | {results['prediction_consistency_percent']:.2f}% |",
        f"| 不一致样本数 | {results['mismatch_count']} |",
        f"| 90% 门槛 | {'通过' if results['hls_threshold_passed'] else '未通过'} |",
        "",
        "## 复现命令",
        "",
    ]
    lines.extend(f"```sh\n{command}\n```\n" for command in commands if command)
    lines.extend(
        [
            "结果文件：",
            "",
            f"- 浮点结果：`{results['float_results']}`",
            f"- HLS 结果：`{results['hls_results']}`",
            f"- 不一致列表：`{results['mismatches']}`",
            "",
        ]
    )
    return "\n".join(lines)


def command_summarize(args: argparse.Namespace) -> int:
    mnist = _read_json(Path(args.mnist_summary))
    blob = _read_json(Path(args.blob_metadata))
    float_summary = _read_json(Path(args.float_summary))
    comparison = _read_json(Path(args.compare_report))
    required_mnist = ("images_path", "images_sha256", "labels_path", "labels_sha256", "sample_count")
    required_blob = ("parameter_source", "parameter_source_sha256", "parameter_count", "sample_count")
    required_comparison = (
        "sample_count", "float_correct", "float_accuracy_percent", "hls_correct",
        "hls_accuracy_percent", "absolute_accuracy_difference_percent",
        "prediction_consistency_percent", "mismatch_count", "hls_threshold_passed",
        "float_results", "hls_results", "mismatches",
    )
    for key in required_mnist:
        if key not in mnist:
            raise ValidationError(f"MNIST summary is missing {key}")
    for key in required_blob:
        if key not in blob:
            raise ValidationError(f"blob metadata is missing {key}")
    for key in required_comparison:
        if key not in comparison:
            raise ValidationError(f"comparison report is missing {key}")
    if mnist["sample_count"] != blob["sample_count"] or blob["sample_count"] != comparison["sample_count"]:
        raise ValidationError("MNIST, blob, and comparison sample counts do not match")
    summary = {
        "status": comparison.get("status", "UNKNOWN"),
        "sample_count": comparison["sample_count"],
        "normalization": NORMALIZATION,
        "fixed_point": {
            "data_t": "ap_fixed<16, 6, AP_RND, AP_SAT>",
            "acc_t": "ap_fixed<32, 14, AP_RND, AP_SAT>",
        },
        "hls_tool_version": args.hls_tool_version,
        "toolchain_status": args.toolchain_status,
        "mnist": mnist,
        "parameters": {
            "source_path": blob["parameter_source"],
            "source_sha256": blob["parameter_source_sha256"],
            "count": blob["parameter_count"],
        },
        "float_summary": float_summary,
        "results": comparison,
        "commands": [args.prepare_command, args.float_command, args.hls_command, args.compare_command],
    }
    _write_json(Path(args.output_json), summary)
    if args.output_markdown:
        markdown_path = Path(args.output_markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if comparison.get("hls_threshold_passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-blob")
    inspect.add_argument("blob", type=Path)
    inspect.add_argument("--expected-count", type=int)
    inspect.add_argument("--json-output", type=Path)
    inspect.set_defaults(function=command_inspect_blob)

    validate = subparsers.add_parser("validate-mnist")
    validate.add_argument("--images", required=True, type=Path)
    validate.add_argument("--labels", required=True, type=Path)
    validate.add_argument("--expected-count", type=int, default=EXPECTED_MNIST_COUNT)
    validate.add_argument("--json-output", type=Path)
    validate.set_defaults(function=command_validate_mnist)

    make_blob = subparsers.add_parser("make-blob")
    make_blob.add_argument("--parameters", required=True, type=Path)
    make_blob.add_argument("--images", required=True, type=Path)
    make_blob.add_argument("--labels", required=True, type=Path)
    make_blob.add_argument("--output", required=True, type=Path)
    make_blob.add_argument("--metadata", type=Path)
    make_blob.add_argument("--expected-count", type=int, default=EXPECTED_MNIST_COUNT)
    make_blob.set_defaults(function=command_make_blob)

    run_float = subparsers.add_parser("run-float")
    run_float.add_argument("--blob", required=True, type=Path)
    run_float.add_argument("--results", required=True, type=Path)
    run_float.add_argument("--summary", type=Path)
    run_float.add_argument("--expected-count", type=int)
    run_float.add_argument("--batch-size", type=int, default=128)
    run_float.set_defaults(function=command_run_float)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--float-results", required=True, type=Path)
    compare.add_argument("--hls-results", required=True, type=Path)
    compare.add_argument("--report", required=True, type=Path)
    compare.add_argument("--mismatches", type=Path)
    compare.add_argument("--threshold", type=float, default=90.0)
    compare.set_defaults(function=command_compare)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--mnist-summary", required=True, type=Path)
    summarize.add_argument("--blob-metadata", required=True, type=Path)
    summarize.add_argument("--float-summary", required=True, type=Path)
    summarize.add_argument("--compare-report", required=True, type=Path)
    summarize.add_argument("--output-json", required=True, type=Path)
    summarize.add_argument("--output-markdown", type=Path)
    summarize.add_argument("--hls-tool-version", required=True)
    summarize.add_argument("--toolchain-status", required=True)
    summarize.add_argument("--prepare-command", default="")
    summarize.add_argument("--float-command", default="")
    summarize.add_argument("--hls-command", default="")
    summarize.add_argument("--compare-command", default="")
    summarize.set_defaults(function=command_summarize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (OSError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
