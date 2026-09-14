#!/usr/bin/env python3
"""Summarize QAT HLS C simulation predictions and logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping


def read_predictions(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["index", "expected", "prediction"] + [f"logit_{i}" for i in range(10)]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected HLS CSV header {reader.fieldnames}")
        for position, row in enumerate(reader):
            index = int(row["index"])
            if index != position:
                raise ValueError(f"bad row index {index}, expected {position}")
            rows.append(
                {
                    "index": index,
                    "expected": int(row["expected"]),
                    "prediction": int(row["prediction"]),
                    "logits": [float(row[f"logit_{i}"]) for i in range(10)],
                }
            )
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def read_software_metrics(path: Path | None) -> Dict[str, Mapping[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        return {row["model"]: row for row in csv.DictReader(stream)}


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def write_report(
    path: Path,
    summary: Mapping[str, object],
    software_metrics: Mapping[str, Mapping[str, str]],
) -> None:
    hls = summary["hls"]
    lines = [
        "# QAT HLS C Simulation",
        "",
        "## 方法",
        "",
        (
            "使用 bonus 导出的 QAT MNIST blob 运行 `level1` 的 LeNet HLS C simulation。"
            "工作目录、输入blob和输出CSV均位于 `bonus/`，未覆盖已有 Level 1 或混合精度结果。"
        ),
        "",
        "## 配置",
        "",
        f"- `LENET_DATA_W`：{summary['data_width']}，即 `ap_fixed<{summary['data_width']},6,AP_RND,AP_SAT>`",
        f"- 输入blob：`{summary['input_blob']}`",
        f"- HLS工作目录：`{summary['hls_workspace']}`",
        f"- 样本数：{hls['sample_count']}",
        f"- HLS返回码：{summary['hls_return_code']}",
        "",
        "## 结果",
        "",
        "|项目|准确率|正确数|预测一致率|备注|",
        "|---|---:|---:|---:|---|",
    ]
    qat_row = software_metrics.get("qat")
    if qat_row:
        lines.append(
            "|QAT软件fake-quant|{acc}|{correct}/{count}|{match}|来自 `bonus/results/w8a8_i6_qat/metrics.csv`|".format(
                acc=format_percent(float(qat_row["accuracy_percent"])),
                correct=qat_row["correct"],
                count=qat_row["sample_count"],
                match=format_percent(float(qat_row["prediction_match_float_percent"])),
            )
        )
    lines.append(
        "|QAT HLS CSim|{acc}|{correct}/{count}||真实 HLS C simulation 输出|".format(
            acc=format_percent(float(hls["accuracy_percent"])),
            correct=hls["correct"],
            count=hls["sample_count"],
        )
    )
    if qat_row:
        delta = float(qat_row["accuracy_percent"]) - float(hls["accuracy_percent"])
        lines += [
            "",
            f"QAT软件fake-quant与HLS CSim准确率差值：{delta:.2f} pp。",
        ]
    lines += [
        "",
        "## 状态",
        "",
        f"- CSim完成：{summary['csim_complete']}",
        f"- `BATCH COMPLETE`：{summary['batch_complete']}",
        f"- 达到准确率阈值 {summary['accuracy_threshold_percent']:.2f}%：{summary['accuracy_passed']}",
        "",
        "## 边界",
        "",
        "本文件只报告 QAT 权重在现有 LeNet HLS C 模型上的功能仿真；未运行综合、RTL协同仿真或资源时序评估。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--software-metrics", type=Path)
    parser.add_argument("--input-blob", type=Path, required=True)
    parser.add_argument("--hls-workspace", type=Path, required=True)
    parser.add_argument("--data-width", type=int, required=True)
    parser.add_argument("--hls-return-code", type=int, required=True)
    parser.add_argument("--accuracy-threshold", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = read_predictions(args.predictions)
    sample_count = len(predictions)
    correct = sum(int(row["expected"] == row["prediction"]) for row in predictions)
    accuracy = 100.0 * correct / sample_count
    log_text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.exists() else ""
    log_accuracy = None
    match = re.search(r"fixed_point_accuracy=([0-9.]+)%", log_text)
    if match:
        log_accuracy = float(match.group(1))
    summary = {
        "data_width": args.data_width,
        "input_blob": str(args.input_blob),
        "hls_workspace": str(args.hls_workspace),
        "predictions_csv": str(args.predictions),
        "run_log": str(args.log),
        "hls_return_code": args.hls_return_code,
        "csim_complete": "CSim done with 0 errors" in log_text,
        "batch_complete": "BATCH COMPLETE" in log_text,
        "accuracy_threshold_percent": args.accuracy_threshold,
        "accuracy_passed": accuracy >= args.accuracy_threshold,
        "hls": {
            "sample_count": sample_count,
            "correct": correct,
            "accuracy_percent": accuracy,
            "log_accuracy_percent": log_accuracy,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    software_metrics = read_software_metrics(args.software_metrics)
    write_report(args.report, summary, software_metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
