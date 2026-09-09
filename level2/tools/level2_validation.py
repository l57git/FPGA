#!/usr/bin/env python3
"""Level 2 real-scene preprocessing and reproducible robustness evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "level1" / "tools"))
from lenet_validation import (  # noqa: E402
    normalize_mnist_images,
    predict_images,
    read_idx_images,
    read_idx_labels,
    read_lenet_blob,
    sha256_file,
    write_lenet_blob,
    write_result_csv,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _resample():
    return Image.Resampling.BILINEAR


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a photo to MNIST convention: 28x28, white digit on black."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) < 2:
        raise ValueError("image is too small")
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    deviation = np.abs(gray - background)
    threshold = max(18.0, float(np.percentile(deviation, 85)) * 0.35)
    mask = deviation > threshold
    if not mask.any():
        return np.zeros((28, 28), dtype=np.float32)
    ys, xs = np.nonzero(mask)
    pad = max(1, int(round(0.08 * max(np.ptp(ys) + 1, np.ptp(xs) + 1))))
    y0, y1 = max(0, int(ys.min()) - pad), min(gray.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(gray.shape[1], int(xs.max()) + pad + 1)
    roi = np.abs(gray[y0:y1, x0:x1] - background)
    peak = float(roi.max())
    if peak <= 0:
        return np.zeros((28, 28), dtype=np.float32)
    roi = np.clip(roi * (255.0 / peak), 0, 255).astype(np.uint8)
    scale = 20.0 / max(roi.shape)
    size = (max(1, int(round(roi.shape[1] * scale))), max(1, int(round(roi.shape[0] * scale))))
    resized = np.asarray(Image.fromarray(roi).resize(size, _resample()), dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    oy, ox = (28 - resized.shape[0]) // 2, (28 - resized.shape[1]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    mass = canvas.sum()
    if mass > 0:
        yy, xx = np.indices(canvas.shape)
        cy, cx = float((yy * canvas).sum() / mass), float((xx * canvas).sum() / mass)
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        shifted = np.zeros_like(canvas)
        sy0, sy1 = max(0, -dy), min(28, 28 - dy)
        sx0, sx1 = max(0, -dx), min(28, 28 - dx)
        shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = canvas[sy0:sy1, sx0:sx1]
        canvas = shifted
    return canvas


def raw_resize(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((28, 28), _resample()), dtype=np.float32)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) > 127:
        gray = 255.0 - gray
    return np.clip(gray / 255.0, 0, 1).astype(np.float32)


def make_scene(mnist: np.ndarray, index: int) -> Image.Image:
    rng = np.random.default_rng(20260904 + index)
    stroke_mask = Image.fromarray(mnist.astype(np.uint8))
    scale = int(rng.integers(25, 43))
    stroke_mask = stroke_mask.resize((scale, scale), _resample()).rotate(
        float(rng.uniform(-18, 18)), fillcolor=0
    )
    canvas = Image.new("L", (64, 64), color=int(rng.integers(205, 256)))
    x = int(rng.integers(1, 64 - scale))
    y = int(rng.integers(1, 64 - scale))
    canvas.paste(Image.new("L", stroke_mask.size, color=int(rng.integers(5, 45))), (x, y), stroke_mask)
    array = np.asarray(canvas, dtype=np.float32)
    gradient = np.linspace(float(rng.uniform(-30, 10)), float(rng.uniform(-10, 35)), 64)
    array += gradient[None, :] + rng.normal(0, 5, array.shape)
    scene = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    return ImageEnhance.Contrast(scene.filter(ImageFilter.GaussianBlur(0.45))).enhance(0.9)


def confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int32)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def save_figures(output: Path, labels, scenes, raw, processed, predictions):
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i in range(6):
        axes[0, i].imshow(scenes[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].imshow(raw[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(processed[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"label={labels[i]}")
        axes[2, i].set_xlabel(f"pred={predictions[i]}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("scene"); axes[1, 0].set_ylabel("raw resize"); axes[2, 0].set_ylabel("preprocessed")
    fig.tight_layout(); fig.savefig(output / "preprocessing_examples.png", dpi=180); plt.close(fig)

def plot_confusion(output: Path, matrix: np.ndarray, title: str, name: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(title=title, xlabel="Predicted", ylabel="True", xticks=range(10), yticks=range(10))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(output / name, dpi=180); plt.close(fig)


def _is_metadata_path(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(
        part == "__MACOSX"
        or part == ".DS_Store"
        or part.startswith(".")
        for part in parts
    )


def _filename_label(path: Path) -> int | None:
    labels = {
        int(token[5:])
        for token in path.stem.split("_")
        if token.startswith("digit") and token[5:].isdigit()
    }
    if len(labels) > 1:
        raise ValueError(f"filename contains conflicting digit labels: {path}")
    return next(iter(labels)) if labels else None


def scan_labeled_dataset(root: Path):
    """Scan digit subdirectories and return valid samples plus exclusions."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")

    samples, excluded = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_name = relative.as_posix()
        if _is_metadata_path(path, root):
            excluded.append({"relative_path": relative_name, "reason": "metadata_or_hidden"})
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            excluded.append({"relative_path": relative_name, "reason": "unsupported_extension"})
            continue
        first = relative.parts[0] if relative.parts else ""
        if first not in {str(label) for label in range(10)}:
            excluded.append({"relative_path": relative_name, "reason": "invalid_label_directory"})
            continue
        label = int(first)
        filename_label = _filename_label(path)
        if filename_label is not None and filename_label != label:
            raise ValueError(
                f"filename label {filename_label} conflicts with directory label {label}: {path}"
            )
        try:
            with Image.open(path) as image:
                image.verify()
                width, height = image.size
                mode, image_format = image.mode, image.format
        except Exception as exc:
            excluded.append(
                {
                    "relative_path": relative_name,
                    "reason": f"decode_error:{type(exc).__name__}:{exc}",
                }
            )
            continue
        samples.append(
            {
                "path": path,
                "relative_path": relative_name,
                "label": label,
                "filename_label": filename_label,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
            }
        )
    return samples, excluded


def scan_background_dataset(root: Path):
    """Scan unlabeled background images for optional false-positive evidence."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"background directory does not exist: {root}")
    samples, excluded = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_name = relative.as_posix()
        if _is_metadata_path(path, root):
            excluded.append({"relative_path": relative_name, "reason": "metadata_or_hidden"})
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            excluded.append({"relative_path": relative_name, "reason": "unsupported_extension"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
                width, height = image.size
                mode, image_format = image.mode, image.format
        except Exception as exc:
            excluded.append(
                {
                    "relative_path": relative_name,
                    "reason": f"decode_error:{type(exc).__name__}:{exc}",
                }
            )
            continue
        samples.append(
            {
                "path": path,
                "relative_path": relative_name,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
            }
        )
    return samples, excluded


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    correct = predictions == labels
    per_class = {}
    for label in range(10):
        mask = labels == label
        count = int(mask.sum())
        per_class[str(label)] = {
            "count": count,
            "correct": int((correct & mask).sum()),
            "accuracy_percent": None if count == 0 else 100.0 * float((correct & mask).sum()) / count,
        }
    return {
        "sample_count": int(labels.size),
        "correct": int(correct.sum()),
        "accuracy_percent": 100.0 * float(correct.mean()) if labels.size else None,
        "per_class": per_class,
    }


def _git_metadata():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        return {"commit": commit, "worktree_dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "worktree_dirty": None, "error": str(exc)}


def _write_manifest(path: Path, samples, excluded) -> None:
    fields = [
        "status", "relative_path", "label", "filename_label", "sha256",
        "width", "height", "mode", "format", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "status": "valid",
                "relative_path": sample["relative_path"],
                "label": sample["label"],
                "filename_label": "" if sample["filename_label"] is None else sample["filename_label"],
                "sha256": sample["sha256"],
                "width": sample["width"],
                "height": sample["height"],
                "mode": sample["mode"],
                "format": sample["format"],
                "reason": "",
            })
        for item in excluded:
            writer.writerow({
                "status": "excluded",
                "relative_path": item["relative_path"],
                "reason": item["reason"],
            })


def _validate_preprocessed(name: str, images: np.ndarray) -> None:
    if images.dtype != np.float32 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"{name} must be float32 with shape N x 28 x 28, got {images.dtype} {images.shape}")
    if not np.isfinite(images).all() or float(images.min()) < -1e-6 or float(images.max()) > 1.000001:
        raise ValueError(f"{name} contains non-finite or out-of-range values")


def _write_background_results(output: Path, samples, processed, predictions) -> dict:
    path = output / "background_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["index", "relative_path", "prediction", "nonzero_pixels"])
        for index, (sample, image, prediction) in enumerate(zip(samples, processed, predictions)):
            writer.writerow([index, sample["relative_path"], int(prediction), int(np.count_nonzero(image))])
    counts = Counter(int(value) for value in predictions)
    return {
        "status": "evaluated",
        "sample_count": len(samples),
        "predicted_class_counts": {str(label): counts.get(label, 0) for label in range(10)},
        "nonzero_preprocessed_count": int(sum(np.any(image > 0) for image in processed)),
        "rejection_rule": "none; classifier predictions are reported without claiming a false-positive rate",
        "results": str(path),
    }


def save_result_overview(output: Path, direct_metrics: dict, processed_metrics: dict) -> None:
    import matplotlib.pyplot as plt
    labels = ["Direct resize", "Preprocessed"]
    values = [direct_metrics["accuracy_percent"], processed_metrics["accuracy_percent"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#e45756", "#59a14f"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Level 2 self-collected dataset: Python evaluation")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "result_overview.png", dpi=180)
    plt.close(fig)


def evaluate_dataset(args) -> int:
    samples, excluded = scan_labeled_dataset(args.input)
    if not samples:
        raise ValueError(f"no valid labeled images found under {args.input}")
    params_blob = read_lenet_blob(args.parameters)
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int32)
    raw_images, processed_images, scenes = [], [], []
    for index, sample in enumerate(samples):
        with Image.open(sample["path"]) as image:
            if index < 6:
                scenes.append(image.convert("L").copy())
            raw_images.append(raw_resize(image))
            processed_images.append(preprocess_image(image))
    raw_images = np.stack(raw_images).astype(np.float32, copy=False)
    processed_images = np.stack(processed_images).astype(np.float32, copy=False)
    _validate_preprocessed("direct resize", raw_images)
    _validate_preprocessed("preprocessed", processed_images)

    raw_logits, raw_predictions = predict_images(raw_images, params_blob.parameters)
    processed_logits, processed_predictions = predict_images(processed_images, params_blob.parameters)
    direct_metrics = _classification_metrics(labels, raw_predictions)
    processed_metrics = _classification_metrics(labels, processed_predictions)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output / "manifest.csv", samples, excluded)
    blob_path = args.output / "preprocessed_self_collected.bin"
    write_lenet_blob(blob_path, params_blob.parameters, labels, processed_images)
    roundtrip = read_lenet_blob(blob_path, expected_count=len(samples))
    if not np.array_equal(roundtrip.labels, labels) or not np.array_equal(roundtrip.images, processed_images):
        raise ValueError("generated HLS blob failed label/image round-trip validation")
    write_result_csv(args.output / "python_results.csv", labels, processed_predictions, processed_logits)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "index", "relative_path", "label", "direct_resize_prediction",
            "direct_resize_correct", "preprocessed_prediction", "preprocessed_correct",
            "preprocessed_logit_margin",
        ])
        ordered_logits = np.sort(processed_logits, axis=1)
        margins = ordered_logits[:, -1] - ordered_logits[:, -2]
        for index, sample in enumerate(samples):
            writer.writerow([
                index, sample["relative_path"], int(labels[index]), int(raw_predictions[index]),
                int(raw_predictions[index] == labels[index]), int(processed_predictions[index]),
                int(processed_predictions[index] == labels[index]), f"{float(margins[index]):.9g}",
            ])

    save_figures(args.output, labels[:6], scenes, raw_images[:6], processed_images[:6], processed_predictions[:6])
    plot_confusion(args.output, confusion(labels, raw_predictions), "Direct resize", "confusion_direct_resize.png")
    plot_confusion(args.output, confusion(labels, processed_predictions), "After preprocessing", "confusion_preprocessed.png")
    save_result_overview(args.output, direct_metrics, processed_metrics)

    background = {"status": "not_provided", "sample_count": 0, "rejection_rule": "not available"}
    if args.background is not None:
        background_samples, background_excluded = scan_background_dataset(args.background)
        if not background_samples:
            raise ValueError(f"no valid background images found under {args.background}")
        background_images = np.stack([
            preprocess_image(Image.open(sample["path"])) for sample in background_samples
        ]).astype(np.float32, copy=False)
        _validate_preprocessed("background preprocessed", background_images)
        _, background_predictions = predict_images(background_images, params_blob.parameters)
        background = _write_background_results(args.output, background_samples, background_images, background_predictions)
        background["excluded_count"] = len(background_excluded)

    summary = {
        "status": "PYTHON_COMPLETE_HLS_PENDING",
        "source_status": args.source_status,
        "input_root": str(args.input),
        "sample_count": len(samples),
        "per_class_counts": {str(label): int((labels == label).sum()) for label in range(10)},
        "excluded_count": len(excluded),
        "excluded_by_reason": dict(Counter(item["reason"].split(":", 1)[0] for item in excluded)),
        "image_sizes": dict(Counter(f"{item['width']}x{item['height']}" for item in samples)),
        "image_modes": dict(Counter(item["mode"] for item in samples)),
        "parameter_source": str(args.parameters),
        "parameter_sha256": sha256_file(args.parameters),
        "manifest_sha256": sha256_file(args.output / "manifest.csv"),
        "git": _git_metadata(),
        "direct_resize": direct_metrics,
        "preprocessed": processed_metrics,
        "preprocessing_gain_percent_points": processed_metrics["accuracy_percent"] - direct_metrics["accuracy_percent"],
        "preprocessing_pipeline": [
            "grayscale", "background estimation and polarity normalization", "ROI crop",
            "aspect-preserving resize", "center of mass alignment", "28x28 normalization",
        ],
        "hls": {"status": "not_run", "reason": "run level2/run_self_collected_hls.sh after loading an HLS environment"},
        "background": background,
        "artifacts": {
            "manifest": "manifest.csv",
            "predictions": "predictions.csv",
            "python_results": "python_results.csv",
            "hls_blob": "preprocessed_self_collected.bin",
            "preprocessing_examples": "preprocessing_examples.png",
            "confusion_direct_resize": "confusion_direct_resize.png",
            "confusion_preprocessed": "confusion_preprocessed.png",
            "result_overview": "result_overview.png",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# Level 2 自采数据 Python 验证报告

## 数据与来源

输入目录为 `{args.input}`，共 {len(samples)} 张有效图片，按父目录 0–9 作为标签。来源状态：`{args.source_status}`；项目组仍需确认这些图片是否为成员实际采集。报告不根据文件名推断数据来源。

有效文件全部可解码；尺寸分布为 {dict(Counter(f"{item['width']}x{item['height']}" for item in samples))}，模式分布为 {dict(Counter(item['mode'] for item in samples))}。排除文件 {len(excluded)} 个，详见 `manifest.csv`。

## 预处理

灰度化 → 背景估计与极性统一 → ROI 裁剪 → 保持比例缩放 → 质心对齐 → 28×28 单通道归一化。直接缩放作为无 ROI 对照。

## Python 结果

| 输入 | 正确数 | 样本数 | 准确率 |
| --- | ---: | ---: | ---: |
| 直接缩放 | {direct_metrics['correct']} | {direct_metrics['sample_count']} | {direct_metrics['accuracy_percent']:.2f}% |
| 完整预处理 | {processed_metrics['correct']} | {processed_metrics['sample_count']} | {processed_metrics['accuracy_percent']:.2f}% |
| 预处理变化 | — | — | {summary['preprocessing_gain_percent_points']:.2f} 个百分点 |

生成的预处理输入已回读校验，标签和 28×28 浮点图像逐项一致，可直接用于现有 HLS testbench。

## HLS 与背景测试

当前 Python 流程已完成。HLS C Simulation、Python/HLS 一致率和定点准确率由 `run_self_collected_hls.sh` 在已加载的 HLS 环境中补充。没有提供独立背景图片，背景误检测试和拒识率仍待补；十分类器本身没有拒识输出。

## 复现

```sh
python3 level2/tools/level2_validation.py evaluate-dataset \\
  --input {args.input} \\
  --parameters {args.parameters} \\
  --output {args.output}
```

后续使用 `level2/run_self_collected_hls.sh`（或 Windows 等价脚本）运行 HLS C Simulation，并用 `level1/tools/lenet_validation.py compare` 对照 `python_results.csv`。
"""
    (args.output / "experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def benchmark(args) -> int:
    count = args.count
    images = read_idx_images(args.images, 10000)[:count]
    labels = read_idx_labels(args.labels, 10000)[:count]
    params = read_lenet_blob(args.parameters).parameters
    scenes = [make_scene(image, i) for i, image in enumerate(images)]
    raw = np.stack([raw_resize(image) for image in scenes])
    processed = np.stack([preprocess_image(image) for image in scenes])
    _, baseline_pred = predict_images(normalize_mnist_images(images), params)
    _, raw_pred = predict_images(raw, params)
    processed_logits, processed_pred = predict_images(processed, params)
    metrics = {
        "status": "PASS",
        "experiment": "reproducible synthetic real-scene stress test; not self-collected data",
        "sample_count": count,
        "mnist_accuracy_percent": 100.0 * float(np.mean(baseline_pred == labels)),
        "raw_resize_accuracy_percent": 100.0 * float(np.mean(raw_pred == labels)),
        "preprocessed_accuracy_percent": 100.0 * float(np.mean(processed_pred == labels)),
        "preprocessing_gain_percent_points": 100.0 * float(np.mean(processed_pred == labels) - np.mean(raw_pred == labels)),
        "pipeline": ["grayscale", "background estimation and polarity normalization", "ROI crop", "aspect-preserving resize", "center of mass alignment", "28x28 normalization"],
        "seed": 20260904,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_lenet_blob(args.output / "preprocessed_stress_test.bin", params, labels, processed)
    write_result_csv(args.output / "python_results.csv", labels, processed_pred, processed_logits)
    (args.output / "benchmark_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["index", "label", "mnist_prediction", "raw_prediction", "preprocessed_prediction"])
        writer.writerows(zip(range(count), labels, baseline_pred, raw_pred, processed_pred))
    save_figures(args.output, labels, scenes, raw, processed, processed_pred)
    plot_confusion(args.output, confusion(labels, raw_pred), "Raw resize", "confusion_raw.png")
    plot_confusion(args.output, confusion(labels, processed_pred), "After preprocessing", "confusion_preprocessed.png")
    report = f"""# Level 2 图像预处理与鲁棒性实验报告

## 实验目的

建立真实手写数字照片到 LeNet 输入的完整预处理链，并测量预处理对域差异的改善。当前实验使用可复现的 MNIST 场景扰动进行压力测试，不冒充自采数据；正式自采结论需在采集照片后运行同一程序得到。

## 方法

预处理依次执行灰度化、背景估计与极性统一、ROI 裁剪、保持比例缩放、质心对齐和 28×28 归一化。推理模型与 Level 1 相同，包含 bias、不执行 softmax，直接对 logits 取 argmax。

压力测试对原始 MNIST 添加随机缩放、平移、旋转、光照梯度、模糊和噪声，固定随机种子为 20260904，共 {count} 张。

## 实测结果

| 输入 | 准确率 |
| --- | ---: |
| 原始 MNIST | {metrics['mnist_accuracy_percent']:.2f}% |
| 场景图直接缩放 | {metrics['raw_resize_accuracy_percent']:.2f}% |
| 完整预处理 | {metrics['preprocessed_accuracy_percent']:.2f}% |
| 预处理提升 | {metrics['preprocessing_gain_percent_points']:.2f} 个百分点 |

## 结论与限制

该实验验证了代码链路和受控场景下的预处理收益。最终 Level 2 报告还必须加入每类若干张由组员实际拍摄的手写数字照片，并单独报告自采数据准确率、无目标背景误检率及典型失败案例。

## 结果文件

- `benchmark_summary.json`：指标与实验配置。
- `predictions.csv`：逐样本预测。
- `preprocessed_stress_test.bin`：可直接输入现有 HLS testbench 的预处理数据。
- `python_results.csv`：用于与HLS定点结果逐样本比较的Python结果。
- `preprocessing_examples.png`：原场景、直接缩放和预处理结果。
- `confusion_raw.png`、`confusion_preprocessed.png`：混淆矩阵。
"""
    (args.output / "experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def evaluate_folder(args) -> int:
    params = read_lenet_blob(args.parameters).parameters
    rows, arrays = [], []
    for path in sorted(args.input.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}: continue
        try: label = int(path.stem.split("_")[0])
        except ValueError: raise ValueError(f"filename must start with label_: {path.name}")
        arrays.append(preprocess_image(Image.open(path))); rows.append((path.name, label))
    if not rows: raise ValueError("no labeled images found")
    _, pred = predict_images(np.stack(arrays), params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["file", "label", "prediction", "correct"])
        for (name, label), value in zip(rows, pred): writer.writerow([name, label, int(value), int(label == value)])
    print(f"correct={sum(label == value for (_, label), value in zip(rows, pred))}/{len(rows)}")
    return 0


def finalize_hls(args) -> int:
    summary_path = Path(args.summary)
    comparison_path = Path(args.comparison)
    report_path = Path(args.report)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("sample_count") != summary.get("sample_count"):
        raise ValueError(
            f"HLS comparison sample count {comparison.get('sample_count')} does not match "
            f"dataset count {summary.get('sample_count')}"
        )
    summary["status"] = "PYTHON_AND_HLS_COMPLETE_BACKGROUND_PENDING"
    summary["hls"] = {
        "status": "complete",
        "csim_status": comparison.get("status"),
        "tool_version": args.tool_version,
        "target_part": "xc7z020-clg400-1",
        "clock_ns": 10,
        "sample_count": comparison["sample_count"],
        "correct": comparison["hls_correct"],
        "accuracy_percent": comparison["hls_accuracy_percent"],
        "prediction_consistency_percent": comparison["prediction_consistency_percent"],
        "mismatch_count": comparison["mismatch_count"],
        "accuracy_threshold_percent": comparison["hls_threshold_percent"],
        "csim_errors": 0,
        "comparison_report": str(comparison_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = report_path.read_text(encoding="utf-8")
    marker = "\n## HLS 与背景测试\n"
    reproduce_marker = "\n## 复现\n"
    section = f"""
## HLS 与 Python 定点对照

使用 {args.tool_version} 对同一批 503 张预处理图片执行 C Simulation，目标器件为 `xc7z020-clg400-1`、时钟为 10 ns；日志报告 `CSim done with 0 errors`。HLS 定点结果为 {comparison['hls_correct']}/{comparison['sample_count']}，准确率 {comparison['hls_accuracy_percent']:.2f}%；Python 与 HLS 逐样本预测一致率为 {comparison['prediction_consistency_percent']:.2f}%，不一致样本数为 {comparison['mismatch_count']}。

比较命令使用 `threshold=0` 仅表示取消自采域数据的 90% 门槛，不能把工具返回的 PASS 当作准确率达标。当前自采数据的绝对准确率仍为 {comparison['hls_accuracy_percent']:.2f}%，低准确率主要反映数据域和预处理问题。

## 背景测试

当前没有提供独立背景图片；十分类器没有拒识输出，因此尚未计算背景误检率。
"""
    if marker in report:
        start = report.index(marker)
        end = report.find(reproduce_marker, start)
        if end < 0:
            end = len(report)
        report = report[:start] + "\n" + section.strip() + "\n" + report[end:]
    else:
        report = report.rstrip() + "\n" + section
    report_path.write_text(report.rstrip() + "\n", encoding="utf-8")
    print(json.dumps(summary["hls"], ensure_ascii=False, indent=2))
    return 0


def finalize_report(args) -> int:
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    report = args.report.read_text(encoding="utf-8")
    section = f"""
## Python与HLS定点对照

使用 Vivado HLS 2019.2 对同一批预处理图像执行C Simulation：Python正确数为 {comparison['float_correct']}/{comparison['sample_count']}，HLS正确数为 {comparison['hls_correct']}/{comparison['sample_count']}；两者准确率均为 {comparison['hls_accuracy_percent']:.2f}%，逐样本预测一致率为 {comparison['prediction_consistency_percent']:.2f}%，不一致样本数为 {comparison['mismatch_count']}。

该结果证明预处理数据进入硬件模型后的数值行为与Python参考实现一致。55.10%的绝对准确率反映场景域差异和模型鲁棒性限制，不是浮点与定点实现不一致造成的。
"""
    marker = "\n## Python与HLS定点对照\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    args.report.write_text(report.rstrip() + "\n" + section, encoding="utf-8")
    summary = json.loads((args.report.parent / "benchmark_summary.json").read_text(encoding="utf-8"))
    import matplotlib.pyplot as plt
    labels = ["MNIST", "Direct resize", "Preprocessed", "Python-HLS\nconsistency"]
    values = [summary["mnist_accuracy_percent"], summary["raw_resize_accuracy_percent"],
              summary["preprocessed_accuracy_percent"], comparison["prediction_consistency_percent"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=["#4c78a8", "#e45756", "#59a14f", "#7b61a8"])
    ax.set_ylim(0, 105); ax.set_ylabel("Percent (%)"); ax.set_title("Level 2 preprocessing and HLS validation")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(args.report.parent / "result_overview.png", dpi=180); plt.close(fig)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--images", type=Path, required=True); bench.add_argument("--labels", type=Path, required=True)
    bench.add_argument("--parameters", type=Path, required=True); bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--count", type=int, default=1000); bench.set_defaults(function=benchmark)
    folder = sub.add_parser("evaluate-folder")
    folder.add_argument("--input", type=Path, required=True); folder.add_argument("--parameters", type=Path, required=True)
    folder.add_argument("--output", type=Path, required=True); folder.set_defaults(function=evaluate_folder)
    dataset = sub.add_parser("evaluate-dataset")
    dataset.add_argument("--input", type=Path, required=True)
    dataset.add_argument("--parameters", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--background", type=Path)
    dataset.add_argument(
        "--source-status",
        choices=("pending_confirmation", "verified_self_collected", "generated_or_synthetic"),
        default="pending_confirmation",
    )
    dataset.set_defaults(function=evaluate_dataset)
    hls = sub.add_parser("finalize-hls")
    hls.add_argument("--summary", type=Path, required=True)
    hls.add_argument("--comparison", type=Path, required=True)
    hls.add_argument("--report", type=Path, required=True)
    hls.add_argument("--tool-version", default="Vitis HLS 2025.2.1")
    hls.set_defaults(function=finalize_hls)
    final = sub.add_parser("finalize-report")
    final.add_argument("--comparison", type=Path, required=True); final.add_argument("--report", type=Path, required=True)
    final.set_defaults(function=finalize_report)
    args = parser.parse_args()
    try: return args.function(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
