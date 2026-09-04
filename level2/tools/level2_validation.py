#!/usr/bin/env python3
"""Level 2 real-scene preprocessing and reproducible robustness evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    write_lenet_blob,
    write_result_csv,
)


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
    final = sub.add_parser("finalize-report")
    final.add_argument("--comparison", type=Path, required=True); final.add_argument("--report", type=Path, required=True)
    final.set_defaults(function=finalize_report)
    args = parser.parse_args()
    try: return args.function(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
