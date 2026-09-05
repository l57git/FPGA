#!/usr/bin/env python3
"""Run and summarize the controlled LeNet fixed-point precision experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
LEVEL1_ROOT = TOOLS_DIR.parent
PROJECT_ROOT = LEVEL1_ROOT.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from lenet_validation import (  # noqa: E402
    EXPECTED_MNIST_COUNT,
    NORMALIZATION,
    PARAMETER_SPECS,
    RESULT_COLUMNS,
    ValidationError,
    _validate_parameters,
    normalize_mnist_images,
    predict_images,
    read_idx_images,
    read_idx_labels,
    read_lenet_blob,
    read_result_csv,
    write_result_csv,
)

WIDTHS = tuple(range(8, 17))
INTEGER_BITS = 6
ACCUMULATOR_TYPE = "ap_fixed<32,14,AP_RND,AP_SAT>"
PART = "xc7z020clg400-1"
CLOCK_NS = 10.0
ACCURACY_THRESHOLD = 90.0
MAX_LOSS_PP = 0.5
EXPERIMENT_NAME = "LeNet shared data_t numerical precision sweep"

SOURCE_FILES = (
    LEVEL1_ROOT / "src" / "lenet.hpp",
    LEVEL1_ROOT / "src" / "lenet.cpp",
    LEVEL1_ROOT / "tb" / "tb_lenet.cpp",
    LEVEL1_ROOT / "run_hls.tcl",
)

SWEEP_COLUMNS = (
    "width",
    "fraction_bits",
    "sample_count",
    "float_accuracy_pct",
    "python_approx_accuracy_pct",
    "hls_accuracy_pct",
    "loss_vs_16_pp",
    "loss_vs_float_pp",
    "agreement_pct",
    "logit_mae",
    "logit_max_abs_error",
    "lut",
    "ff",
    "dsp",
    "bram_18k",
    "latency_cycles_max",
    "latency_ns_target",
    "estimated_clock_ns",
    "csim_status",
    "synth_status",
    "parse_status",
    "eligible",
    "error",
)


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_read(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_width(width: int) -> None:
    if width not in WIDTHS:
        raise ValidationError(f"data width must be one of 8..16, got {width}")


def quantize(values: np.ndarray | float, width: int) -> np.ndarray:
    """Approximate AP_RND/AP_SAT at a data_t assignment boundary.

    AP_RND's half-way behavior is represented explicitly as floor(x + 0.5),
    which sends an exact negative half step toward positive infinity.
    """

    _validate_width(int(width))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValidationError("cannot quantize non-finite values")
    fraction_bits = int(width) - INTEGER_BITS
    scale = float(1 << fraction_bits)
    minimum = -(1 << (int(width) - 1))
    maximum = (1 << (int(width) - 1)) - 1
    integer_values = np.floor(array * scale + 0.5)
    integer_values = np.clip(integer_values, minimum, maximum)
    return integer_values / scale


def _range_stats(values: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValidationError("range input is empty or non-finite")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "max_abs": float(np.max(np.abs(array))),
    }


def _merge_range(
    ranges: MutableMapping[str, Dict[str, Any]], name: str, values: np.ndarray
) -> None:
    current = _range_stats(values)
    if name not in ranges:
        ranges[name] = current
        return
    previous = ranges[name]
    previous["count"] += current["count"]
    previous["min"] = min(previous["min"], current["min"])
    previous["max"] = max(previous["max"], current["max"])
    previous["max_abs"] = max(previous["max_abs"], current["max_abs"])


def _conv2d_valid64(
    inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(inputs, dtype=np.float64),
        (weights.shape[2], weights.shape[3]),
        axis=(-2, -1),
    )
    output = np.einsum(
        "ncyxkl,ockl->noyx",
        windows,
        np.asarray(weights, dtype=np.float64),
        optimize=True,
    )
    output += np.asarray(biases, dtype=np.float64)[None, :, None, None]
    return output


def _max_pool64(inputs: np.ndarray) -> np.ndarray:
    array = np.asarray(inputs, dtype=np.float64)
    rows = array.shape[2] // 2
    columns = array.shape[3] // 2
    return array.reshape(array.shape[0], array.shape[1], rows, 2, columns, 2).max(
        axis=(3, 5)
    )


def _empty_ranges() -> Dict[str, Any]:
    return {"input": None, "parameters": {}, "layers": {}}


def collect_float_ranges(
    images: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Collect ranges from the existing float32 layer ordering."""

    _validate_parameters(dict(parameters))
    result = _empty_ranges()
    result["input"] = _range_stats(images)
    for name, _, _ in PARAMETER_SPECS:
        result["parameters"][name] = _range_stats(parameters[name])

    images = np.asarray(images, dtype=np.float32)
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        inputs = images[start:end, None, :, :]
        conv1 = _conv2d_valid64(
            inputs,
            np.asarray(parameters["conv1_w"][:, None, :, :], dtype=np.float64),
            parameters["conv1_b"],
        ).astype(np.float32)
        _merge_range(result["layers"], "conv1_pre_quant", conv1)
        pool1 = _max_pool64(np.maximum(conv1, np.float32(0.0)))
        conv2 = _conv2d_valid64(
            pool1,
            parameters["conv2_w"],
            parameters["conv2_b"],
        ).astype(np.float32)
        _merge_range(result["layers"], "conv2_pre_quant", conv2)
        pool2 = _max_pool64(np.maximum(conv2, np.float32(0.0)))
        flattened = pool2.reshape(pool2.shape[0], -1)
        fc1 = (
            flattened @ parameters["fc1_w"].T + parameters["fc1_b"]
        ).astype(np.float32)
        _merge_range(result["layers"], "fc1_pre_quant", fc1)
        fc1 = np.maximum(fc1, np.float32(0.0))
        fc2 = (fc1 @ parameters["fc2_w"].T + parameters["fc2_b"]).astype(np.float32)
        _merge_range(result["layers"], "fc2_pre_quant", fc2)
        fc2 = np.maximum(fc2, np.float32(0.0))
        fc3 = (fc2 @ parameters["fc3_w"].T + parameters["fc3_b"]).astype(np.float32)
        _merge_range(result["layers"], "fc3_pre_quant", fc3)
    return result


def python_approx_forward(
    images: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    width: int,
    batch_size: int = 128,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the explicitly named layer-boundary quantization approximation."""

    _validate_width(width)
    _validate_parameters(dict(parameters))
    quantized_parameters = {
        name: quantize(parameters[name], width) for name, _, _ in PARAMETER_SPECS
    }
    result_ranges = _empty_ranges()
    result_ranges["input"] = _range_stats(quantize(images, width))
    for name in quantized_parameters:
        result_ranges["parameters"][name] = _range_stats(quantized_parameters[name])

    images = np.asarray(images, dtype=np.float64)
    logits = np.empty((images.shape[0], 10), dtype=np.float64)
    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        inputs = quantize(images[start:end, None, :, :], width)
        conv1 = _conv2d_valid64(
            inputs,
            quantized_parameters["conv1_w"][:, None, :, :],
            quantized_parameters["conv1_b"],
        )
        _merge_range(result_ranges["layers"], "conv1_pre_quant", conv1)
        conv1 = np.maximum(quantize(conv1, width), 0.0)
        pool1 = _max_pool64(conv1)
        conv2 = _conv2d_valid64(
            pool1,
            quantized_parameters["conv2_w"],
            quantized_parameters["conv2_b"],
        )
        _merge_range(result_ranges["layers"], "conv2_pre_quant", conv2)
        conv2 = np.maximum(quantize(conv2, width), 0.0)
        pool2 = _max_pool64(conv2)
        flattened = pool2.reshape(pool2.shape[0], -1)
        fc1 = flattened @ quantized_parameters["fc1_w"].T + quantized_parameters["fc1_b"]
        _merge_range(result_ranges["layers"], "fc1_pre_quant", fc1)
        fc1 = np.maximum(quantize(fc1, width), 0.0)
        fc2 = fc1 @ quantized_parameters["fc2_w"].T + quantized_parameters["fc2_b"]
        _merge_range(result_ranges["layers"], "fc2_pre_quant", fc2)
        fc2 = np.maximum(quantize(fc2, width), 0.0)
        fc3 = fc2 @ quantized_parameters["fc3_w"].T + quantized_parameters["fc3_b"]
        _merge_range(result_ranges["layers"], "fc3_pre_quant", fc3)
        logits[start:end] = quantize(fc3, width)
    return logits, result_ranges


def _resolve_metadata_path(value: str, metadata_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    candidates = (
        metadata_path.parent / candidate,
        LEVEL1_ROOT / candidate,
        PROJECT_ROOT / candidate,
        Path.cwd() / candidate,
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return (LEVEL1_ROOT / candidate).resolve()


def _validate_blob_metadata(blob_path: Path, metadata_path: Path) -> Dict[str, Any]:
    metadata = _json_read(metadata_path)
    required = (
        "output_sha256",
        "sample_count",
        "parameter_count",
        "parameter_source",
        "parameter_source_sha256",
        "images_path",
        "images_sha256",
        "labels_path",
        "labels_sha256",
        "normalization",
        "protocol_magic",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValidationError(f"blob metadata is missing: {', '.join(missing)}")
    if metadata["sample_count"] != EXPECTED_MNIST_COUNT:
        raise ValidationError("precision experiment requires sample_count=10000")
    expected_parameter_count = sum(count for _, count, _ in PARAMETER_SPECS)
    if metadata["parameter_count"] != expected_parameter_count:
        raise ValidationError(
            f"metadata parameter_count is {metadata['parameter_count']}, "
            f"expected {expected_parameter_count}"
        )
    if metadata["normalization"] != NORMALIZATION:
        raise ValidationError(
            f"metadata normalization is {metadata['normalization']!r}, expected {NORMALIZATION!r}"
        )
    if metadata["protocol_magic"] != -20260902:
        raise ValidationError("metadata protocol_magic does not match the blob protocol")
    actual_blob_hash = _sha256_file(blob_path)
    if metadata["output_sha256"] != actual_blob_hash:
        raise ValidationError(
            f"blob SHA-256 does not match metadata: {actual_blob_hash} != {metadata['output_sha256']}"
        )

    images_path = _resolve_metadata_path(str(metadata["images_path"]), metadata_path)
    labels_path = _resolve_metadata_path(str(metadata["labels_path"]), metadata_path)
    parameter_source = _resolve_metadata_path(
        str(metadata["parameter_source"]), metadata_path
    )
    for path, key in (
        (images_path, "images_sha256"),
        (labels_path, "labels_sha256"),
        (parameter_source, "parameter_source_sha256"),
    ):
        if not path.is_file():
            raise ValidationError(f"metadata source is not available: {path}")
        actual = _sha256_file(path)
        if actual != metadata[key]:
            raise ValidationError(
                f"source SHA-256 does not match metadata for {path}: {actual} != {metadata[key]}"
            )

    blob = read_lenet_blob(blob_path, EXPECTED_MNIST_COUNT)
    dataset_images = read_idx_images(images_path, EXPECTED_MNIST_COUNT)
    dataset_labels = read_idx_labels(labels_path, EXPECTED_MNIST_COUNT)
    normalized_images = normalize_mnist_images(dataset_images)
    if not np.array_equal(blob.labels, dataset_labels):
        raise ValidationError("blob labels do not match the official MNIST labels")
    if not np.array_equal(blob.images, normalized_images):
        raise ValidationError("blob images do not match uint8 / 255.0 MNIST normalization")

    source_blob = read_lenet_blob(parameter_source)
    for name, _, _ in PARAMETER_SPECS:
        if not np.array_equal(blob.parameters[name], source_blob.parameters[name]):
            raise ValidationError(f"blob parameter {name} differs from parameter source")
    metadata["resolved_images_path"] = str(images_path)
    metadata["resolved_labels_path"] = str(labels_path)
    metadata["resolved_parameter_source"] = str(parameter_source)
    metadata["actual_output_sha256"] = actual_blob_hash
    metadata["sample_count"] = EXPECTED_MNIST_COUNT
    return metadata


def _git_metadata() -> Dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run_git("status", "--porcelain", "--untracked-files=all")
    return {
        "revision": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status or "",
    }


def _tool_path(value: str) -> str:
    path = Path(value)
    if path.is_file():
        return str(path.resolve())
    resolved = shutil.which(value)
    if resolved:
        return str(Path(resolved).resolve())
    raise ValidationError(f"HLS executable not found: {value}")


def _hls_command_prefix(executable: str) -> List[str]:
    """Return an argv prefix that also supports the installed Xilinx loader."""

    path = Path(executable)
    if path.name == "loader":
        return [executable, "-exec", "vitis_hls"]
    if path.name == "vitis_hls":
        # Some Vitis installations ship only the unwrapped executable.  The
        # sibling loader supplies RDI_DATADIR/TCL_LIBRARY/LD_LIBRARY_PATH.
        loader = path.parents[2] / "loader" if len(path.parents) > 2 else None
        if loader is not None and loader.is_file():
            return [str(loader), "-exec", "vitis_hls"]
    return [executable]


def _probe_tool(executable: str) -> Dict[str, Any]:
    prefix = _hls_command_prefix(executable)
    for flag in ("-version", "--version"):
        try:
            result = subprocess.run(
                [*prefix, flag],
                cwd=LEVEL1_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"command": [*prefix, flag], "status": "error", "error": str(exc)}
        output = (result.stdout + result.stderr).strip()
        if output:
            return {
                "command": [*prefix, flag],
                "status": "ok" if result.returncode == 0 else "reported",
                "returncode": result.returncode,
                "version": output[-4000:],
            }
    return {"command": [*prefix, "-version"], "status": "unavailable"}


def _make_manifest(
    blob_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    output: Path,
    mode: str,
    hls_executable: str | None,
) -> Dict[str, Any]:
    raw = blob_path.read_bytes()
    parameter_bytes = sum(count for _, count, _ in PARAMETER_SPECS) * 4
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": mode,
        "created": __import__("datetime").datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output": str(output),
        "blob": {
            "path": str(blob_path),
            "sha256": _sha256_bytes(raw),
            "parameter_region_sha256": _sha256_bytes(raw[8 : 8 + parameter_bytes]),
            "sample_count": EXPECTED_MNIST_COUNT,
            "parameter_count": sum(count for _, count, _ in PARAMETER_SPECS),
            "metadata_path": str(metadata_path),
        },
        "mnist": {
            "images_path": metadata["resolved_images_path"],
            "images_sha256": metadata["images_sha256"],
            "labels_path": metadata["resolved_labels_path"],
            "labels_sha256": metadata["labels_sha256"],
            "normalization": NORMALIZATION,
        },
        "parameter_source": {
            "path": metadata["resolved_parameter_source"],
            "sha256": metadata["parameter_source_sha256"],
        },
        "scan": {
            "widths": list(WIDTHS),
            "integer_bits": INTEGER_BITS,
            "fraction_bits": "W-6",
            "data_type": "ap_fixed<W,6,AP_RND,AP_SAT>",
            "accumulator_type": ACCUMULATOR_TYPE,
            "part": PART,
            "clock_ns": CLOCK_NS,
            "accuracy_threshold_pct": ACCURACY_THRESHOLD,
            "max_loss_vs_16_pp": MAX_LOSS_PP,
        },
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
            for path in SOURCE_FILES
        },
        "git": _git_metadata(),
        "toolchain": {
            "requested_executable": hls_executable,
            "status": "python-only" if mode == "python-only" else "pending",
        },
        "commands": [],
        "variants": {},
    }


def _write_variant_state(variant: Path, record: Mapping[str, Any]) -> None:
    _json_write(variant / "metrics.json", dict(record))


def _read_result_arrays(
    path: Path, labels: np.ndarray, expected_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_result_csv(path)
    if len(rows) != expected_count:
        raise ValidationError(
            f"result file {path} has {len(rows)} rows, expected {expected_count}"
        )
    predictions = np.empty(expected_count, dtype=np.int32)
    logits = np.empty((expected_count, 10), dtype=np.float64)
    for index, row in enumerate(rows):
        if int(row["index"]) != index:
            raise ValidationError(f"result index mismatch at row {index} in {path}")
        if int(row["expected"]) != int(labels[index]):
            raise ValidationError(f"result label mismatch at sample {index} in {path}")
        predictions[index] = int(row["prediction"])
        logits[index] = np.asarray(row["logits"], dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValidationError(f"result logits are non-finite: {path}")
    return predictions, logits


def _accuracy(predictions: np.ndarray | None, labels: np.ndarray) -> float | None:
    if predictions is None:
        return None
    return 100.0 * float(np.count_nonzero(predictions == labels)) / labels.size


def _find_csynth_report(workspace: Path) -> Path | None:
    candidates = sorted(
        [path for path in workspace.rglob("*csynth*.xml") if path.is_file()]
        + [path for path in workspace.rglob("*csynth*.rpt") if path.is_file()]
    )
    if not candidates:
        return None
    candidates.sort(key=lambda path: (0 if path.suffix == ".xml" else 1, len(path.parts)))
    return candidates[0]


def _number(text: str) -> float | int | None:
    cleaned = text.strip().replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if value.is_integer():
        return int(value)
    return value


def _xml_metric(root: ET.Element, names: Iterable[str]) -> float | int | None:
    names = set(names)
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name in names and element.text:
            value = _number(element.text)
            if value is not None:
                return value
    return None


def _text_metric(text: str, patterns: Iterable[str]) -> float | int | None:
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for line in text.splitlines():
        if not any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
            continue
        values = re.findall(number_pattern, line.replace(",", ""))
        for value in values:
            parsed = _number(value)
            if parsed is not None:
                return parsed
    return None


def parse_csynth_report(path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "lut": None,
        "ff": None,
        "dsp": None,
        "bram_18k": None,
        "latency_cycles_max": None,
        "estimated_clock_ns": None,
        "report_path": str(path),
    }
    text = path.read_text(encoding="utf-8", errors="replace")
    root: ET.Element | None = None
    if path.suffix.lower() == ".xml":
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            root = None
    if root is not None:
        metrics["lut"] = _xml_metric(root, ("LUT", "LUTs"))
        metrics["ff"] = _xml_metric(root, ("FF", "FFs"))
        metrics["dsp"] = _xml_metric(root, ("DSP48E", "DSP48", "DSP"))
        metrics["bram_18k"] = _xml_metric(root, ("BRAM_18K", "BRAM18K", "BRAM"))
        metrics["latency_cycles_max"] = _xml_metric(
            root, ("LatencyWorst", "WorstLatency", "LatencyMax")
        )
        metrics["estimated_clock_ns"] = _xml_metric(
            root,
            ("EstimatedClockPeriod", "EstimatedClockPeriodNs", "ClockPeriod"),
        )
    if metrics["lut"] is None:
        metrics["lut"] = _text_metric(text, (r"\bLUT\b",))
    if metrics["ff"] is None:
        metrics["ff"] = _text_metric(text, (r"\bFF\b", r"Flip[- ]?Flop"))
    if metrics["dsp"] is None:
        metrics["dsp"] = _text_metric(text, (r"DSP48", r"\bDSP\b"))
    if metrics["bram_18k"] is None:
        metrics["bram_18k"] = _text_metric(text, (r"BRAM_18K", r"BRAM18K"))
    if metrics["latency_cycles_max"] is None:
        metrics["latency_cycles_max"] = _text_metric(
            text, (r"Latency.*Worst", r"Worst.*Latency", r"Latency \(cycles\)")
        )
    if metrics["estimated_clock_ns"] is None:
        metrics["estimated_clock_ns"] = _text_metric(
            text, (r"Estimated.*Clock", r"Clock.*Period")
        )
    required = (
        "lut",
        "ff",
        "dsp",
        "bram_18k",
        "latency_cycles_max",
        "estimated_clock_ns",
    )
    metrics["parse_status"] = "ok" if all(metrics[key] is not None for key in required) else "failed"
    metrics["latency_ns_target"] = (
        float(metrics["latency_cycles_max"]) * CLOCK_NS
        if metrics["latency_cycles_max"] is not None
        else None
    )
    if metrics["parse_status"] != "ok":
        missing = [key for key in required if metrics[key] is None]
        metrics["error"] = "missing report metrics: " + ", ".join(missing)
    return metrics


def _copy_csynth_report(report: Path, variant: Path) -> str:
    suffix = report.suffix.lower()
    target = variant / ("csynth_report.xml" if suffix == ".xml" else "csynth_report.rpt")
    if report.resolve() != target.resolve():
        shutil.copy2(report, target)
    return str(target)


def _run_hls_variant(
    width: int,
    blob_path: Path,
    variant: Path,
    executable: str,
    labels: np.ndarray,
    manifest: MutableMapping[str, Any],
) -> Dict[str, Any]:
    hls_csv = variant / "hls.csv"
    requested_workspace = variant / "hls_work"
    workspace = requested_workspace
    if " " in str(workspace):
        workspace = Path(tempfile.mkdtemp(prefix=f"lenet_precision_w{width}_"))
    command = [*_hls_command_prefix(executable), "-f", str(LEVEL1_ROOT / "run_hls.tcl")]
    env = os.environ.copy()
    env.update(
        {
            "LENET_DATA_W": str(width),
            "LENET_HLS_WORKSPACE": str(workspace),
            "LENET_ACCURACY_BLOB": str(blob_path),
            "LENET_RESULT_CSV": str(hls_csv),
            "LENET_ACCURACY_THRESHOLD": "0",
            "LENET_SKIP_CSIM": "0",
            "LENET_SKIP_SYNTH": "0",
            "LENET_CSIM_OPT": "1",
            "LENET_CSIM_MFLAGS": "CCFLAG+=-O3",
        }
    )
    manifest["commands"].append({"width": width, "argv": command, "cwd": str(LEVEL1_ROOT)})
    variant.mkdir(parents=True, exist_ok=True)
    log_path = variant / "hls.log"
    try:
        result = subprocess.run(
            command,
            cwd=LEVEL1_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        log_text = result.stdout + result.stderr
        returncode = result.returncode
    except OSError as exc:
        log_text = f"cannot execute HLS: {exc}\n"
        returncode = 127
    log_path.write_text(log_text, encoding="utf-8", errors="replace")

    record: Dict[str, Any] = {
        "width": width,
        "fraction_bits": width - INTEGER_BITS,
        "hls_command": command,
        "hls_returncode": returncode,
        "hls_log": str(log_path),
        "hls_csv": str(hls_csv),
        "hls_workspace": str(requested_workspace),
        "hls_workspace_actual": str(workspace),
        "csim_status": "failed",
        "synth_status": "skipped",
        "parse_status": "skipped",
        "error": None,
    }
    try:
        _read_result_arrays(hls_csv, labels, EXPECTED_MNIST_COUNT)
        record["csim_status"] = "ok"
    except (OSError, ValidationError) as exc:
        record["error"] = f"CSim result invalid: {exc}"

    report = _find_csynth_report(workspace)
    if record["csim_status"] == "ok":
        if report is None:
            record["synth_status"] = "failed"
            record["parse_status"] = "failed"
            record["error"] = (record["error"] + "; " if record["error"] else "") + "csynth report not found"
        else:
            record["synth_status"] = "ok" if returncode == 0 else "failed"
            copied_report = _copy_csynth_report(report, variant)
            parsed = parse_csynth_report(Path(copied_report))
            record.update(parsed)
            record["report_path"] = copied_report
            if record["synth_status"] != "ok":
                record["error"] = (record["error"] + "; " if record["error"] else "") + "HLS process returned non-zero"
    else:
        record["synth_status"] = "skipped"
        if returncode != 0:
            record["error"] = (record["error"] + "; " if record["error"] else "") + f"HLS process returned {returncode}"
    if record["csim_status"] == "ok" and returncode != 0 and record["synth_status"] == "skipped":
        record["synth_status"] = "failed"
    if workspace != requested_workspace and workspace.exists() and not requested_workspace.exists():
        shutil.move(str(workspace), str(requested_workspace))
    return record


def _base_record(width: int) -> Dict[str, Any]:
    return {
        "width": width,
        "fraction_bits": width - INTEGER_BITS,
        "sample_count": EXPECTED_MNIST_COUNT,
        "float_accuracy_pct": None,
        "python_approx_accuracy_pct": None,
        "python_approx_agreement_pct": None,
        "hls_accuracy_pct": None,
        "loss_vs_16_pp": None,
        "loss_vs_float_pp": None,
        "agreement_pct": None,
        "logit_mae": None,
        "logit_max_abs_error": None,
        "lut": None,
        "ff": None,
        "dsp": None,
        "bram_18k": None,
        "latency_cycles_max": None,
        "latency_ns_target": None,
        "estimated_clock_ns": None,
        "csim_status": "pending",
        "synth_status": "pending",
        "parse_status": "pending",
        "eligible": False,
        "error": None,
    }


def _finish_records(
    records: List[Dict[str, Any]],
    float_accuracy: float,
    labels: np.ndarray,
    float_predictions: np.ndarray,
    float_logits: np.ndarray,
    mode: str,
) -> Dict[str, Any]:
    by_width = {int(record["width"]): record for record in records}
    baseline = by_width.get(16, {})
    baseline_accuracy = baseline.get("hls_accuracy_pct")
    evidence_complete = mode == "full" and all(
        record.get("csim_status") == "ok"
        and record.get("synth_status") == "ok"
        and record.get("parse_status") == "ok"
        for record in records
    )
    baseline_ok = (
        evidence_complete
        and baseline_accuracy is not None
        and float(baseline_accuracy) >= ACCURACY_THRESHOLD
    )
    for record in records:
        width = int(record["width"])
        record["float_accuracy_pct"] = float_accuracy
        if baseline_accuracy is not None and record.get("hls_accuracy_pct") is not None:
            record["loss_vs_16_pp"] = float(baseline_accuracy) - float(record["hls_accuracy_pct"])
        if record.get("hls_accuracy_pct") is not None:
            record["loss_vs_float_pp"] = float_accuracy - float(record["hls_accuracy_pct"])
        own_eligible = (
            baseline_ok
            and record.get("hls_accuracy_pct") is not None
            and float(record["hls_accuracy_pct"]) >= ACCURACY_THRESHOLD
            and float(record["loss_vs_16_pp"]) <= MAX_LOSS_PP
        )
        record["eligible"] = bool(own_eligible)
    candidates = [int(record["width"]) for record in records if record["eligible"]]
    recommendation = min(candidates) if candidates else None
    reasons: List[str] = []
    if mode == "python-only":
        reasons.append("显式 --python-only 只完成软件近似，缺少 HLS CSim 与综合证据")
    elif not evidence_complete:
        reasons.append("九档 HLS CSim、综合或综合报告解析未全部完成")
    if baseline_accuracy is None:
        reasons.append("16 位 HLS 基线缺失")
    elif float(baseline_accuracy) < ACCURACY_THRESHOLD:
        reasons.append(f"16 位 HLS 基线低于 {ACCURACY_THRESHOLD:.0f}%")
    if not reasons and recommendation is None:
        reasons.append("没有同时满足准确率和相对 16 位损失门槛的位宽")
    if not reasons:
        reasons.append("所有九档证据完整，取满足门槛的最小位宽")
    return {
        "status": "COMPLETE" if evidence_complete else "INCOMPLETE",
        "mode": mode,
        "sample_count": EXPECTED_MNIST_COUNT,
        "float_accuracy_pct": float_accuracy,
        "accuracy_threshold_pct": ACCURACY_THRESHOLD,
        "max_loss_vs_16_pp": MAX_LOSS_PP,
        "recommendation_width": recommendation,
        "recommendation_reason": "；".join(reasons),
        "baseline_width": 16,
        "baseline_hls_accuracy_pct": baseline_accuracy,
        "variants": records,
    }


def _write_sweep_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["width"])):
            writer.writerow({key: record.get(key) for key in SWEEP_COLUMNS})


def _format_value(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_plots(output: Path, records: Sequence[Mapping[str, Any]], float_accuracy: float) -> None:
    mpl_config = Path("/tmp/lenet_precision_mplconfig")
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(records, key=lambda item: int(item["width"]))
    widths = [int(item["width"]) for item in ordered]
    hls = [item.get("hls_accuracy_pct") for item in ordered]
    approx = [item.get("python_approx_accuracy_pct") for item in ordered]
    plt.figure(figsize=(8, 5))
    plt.plot(widths, approx, "o-", label="Python approx")
    plt.plot(widths, hls, "s-", label="HLS CSim")
    plt.axhline(float_accuracy, color="black", linestyle="--", label="Float reference")
    plt.xlabel("data_t width (bits)")
    plt.ylabel("Accuracy (%)")
    plt.xticks(widths)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "accuracy_vs_width.png", dpi=160)
    plt.close()

    resources = (("lut", "LUT"), ("ff", "FF"), ("dsp", "DSP"), ("bram_18k", "BRAM 18K"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, (key, label) in zip(axes.flat, resources):
        axis.plot(widths, [item.get(key) for item in ordered], "o-")
        axis.set_title(label)
        axis.set_xticks(widths)
        axis.grid(True, alpha=0.3)
    fig.supxlabel("data_t width (bits)")
    fig.suptitle("HLS synthesis resources")
    fig.tight_layout()
    fig.savefig(output / "resources_vs_width.png", dpi=160)
    plt.close(fig)


def _write_report_markdown(output: Path, manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    rows = []
    for record in sorted(summary["variants"], key=lambda item: int(item["width"])):
        rows.append(
            "| {width} | {approx} | {hls} | {loss} | {agreement} | {lut} | {ff} | {dsp} | {bram} | {latency} | {status} |".format(
                width=record["width"],
                approx=_format_value(record.get("python_approx_accuracy_pct")),
                hls=_format_value(record.get("hls_accuracy_pct")),
                loss=_format_value(record.get("loss_vs_16_pp")),
                agreement=_format_value(record.get("agreement_pct")),
                lut=_format_value(record.get("lut"), 0),
                ff=_format_value(record.get("ff"), 0),
                dsp=_format_value(record.get("dsp"), 0),
                bram=_format_value(record.get("bram_18k"), 0),
                latency=_format_value(record.get("latency_ns_target"), 1),
                status=("可选" if record.get("eligible") else "—"),
            )
        )
    tool = manifest.get("toolchain", {})
    baseline = next((item for item in summary["variants"] if int(item["width"]) == 16), None)
    hls_records = [
        item for item in summary["variants"] if item.get("hls_accuracy_pct") is not None
    ]
    lowest_hls = min(hls_records, key=lambda item: float(item["hls_accuracy_pct"])) if hls_records else None
    recommended = next(
        (
            item
            for item in summary["variants"]
            if summary.get("recommendation_width") is not None
            and int(item["width"]) == int(summary["recommendation_width"])
        ),
        None,
    )
    lines = [
        "# LeNet 定点位宽/数值精度实验报告",
        "",
        f"- 状态：**{summary['status']}**",
        f"- 模式：`{summary['mode']}`",
        f"- 样本数：`{summary['sample_count']}`（官方 MNIST test split）",
        f"- 输入归一化：`{NORMALIZATION}`",
        f"- 数据类型：`ap_fixed<W,6,AP_RND,AP_SAT>`，W = 8..16",
        f"- 累加器：`{ACCUMULATOR_TYPE}`",
        f"- 器件/目标周期：`{PART}` / `{CLOCK_NS:g} ns`",
        f"- HLS 工具：`{tool.get('version', tool.get('requested_executable', '未记录'))}`",
        "",
        "## 结论",
        "",
        f"- 浮点参考准确率：**{summary['float_accuracy_pct']:.2f}%**",
        f"- 16 位 HLS 基线：**{_format_value(summary.get('baseline_hls_accuracy_pct'))}%**",
        f"- 规则：HLS 准确率 ≥ {ACCURACY_THRESHOLD:.0f}%，相对 16 位下降 ≤ {MAX_LOSS_PP:.2f} 个百分点",
        f"- 推荐位宽：**{summary.get('recommendation_width') if summary.get('recommendation_width') is not None else '无（证据不完整或无可用配置）'}**",
        f"- 说明：{summary['recommendation_reason']}",
        "",
        "Python approx 是层边界量化近似，不是逐 MAC 位精确模型；位宽选择只使用 HLS CSim 结果。",
        "延迟是综合报告的最坏周期数乘以目标周期，资源和时序是综合估计，不是上板实测。",
        "",
    ]
    if lowest_hls is not None:
        lines.append(
            f"- 低位宽端的损失符合固定小数位预算的预期：W={lowest_hls['width']} 只有 "
            f"{int(lowest_hls['width']) - INTEGER_BITS} 个小数位，HLS 准确率为 "
            f"{_format_value(lowest_hls.get('hls_accuracy_pct'))}%；整数位固定为 {INTEGER_BITS}，"
            "因此本实验没有通过动态重缩放来掩盖量化误差。"
        )
    if recommended is not None and baseline is not None:
        changes = []
        resource_labels = {
            "lut": "LUT",
            "ff": "FF",
            "dsp": "DSP",
            "bram_18k": "BRAM18K",
        }
        for key, label in resource_labels.items():
            base = baseline.get(key)
            current = recommended.get(key)
            if base and current is not None:
                change = (float(base) - float(current)) / float(base) * 100.0
                action = "节省" if change >= 0 else "增加"
                changes.append(f"{label}{action}{abs(change):.2f}%")
        if changes:
            lines.append(
                f"- 推荐 W={recommended['width']}：HLS 准确率 "
                f"{_format_value(recommended.get('hls_accuracy_pct'))}%，相对 16 位损失 "
                f"{_format_value(recommended.get('loss_vs_16_pp'))} 个百分点；资源上 "
                + "、".join(changes)
                + "，延迟保持在同一综合估计量级。"
            )
    lines.extend(
        [
            "",
            "## 九档结果",
            "",
            "| W | Python approx % | HLS % | 相对16位损失 pp | HLS一致率 % | LUT | FF | DSP | BRAM18K | 延迟 ns | 选型 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
            *rows,
            "",
            "## 资源相对 16 位变化",
            "",
            "节省百分比按 `(R16 - RW) / R16 * 100` 计算；16 位资源为 0 时记为不可计算。",
            "",
        ]
    )
    if baseline is not None:
        lines.extend(["| W | LUT | FF | DSP | BRAM18K |", "|---:|---:|---:|---:|---:|"])
        for record in sorted(summary["variants"], key=lambda item: int(item["width"])):
            values = []
            for key in ("lut", "ff", "dsp", "bram_18k"):
                base = baseline.get(key)
                current = record.get(key)
                values.append(
                    "不可计算" if not base else _format_value((float(base) - float(current)) / float(base) * 100.0)
                    if current is not None else "—"
                )
            lines.append(f"| {record['width']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 复现信息",
            "",
            f"- blob SHA-256：`{manifest['blob']['sha256']}`",
            f"- 参数区 SHA-256：`{manifest['blob']['parameter_region_sha256']}`",
            f"- Git revision：`{manifest.get('git', {}).get('revision')}`；dirty：`{manifest.get('git', {}).get('dirty')}`",
            "- 原始逐样本结果保存在各 `w*/` 目录，可用 `report` 离线重新汇总。",
            "- 当前环境为 Vitis HLS 开发工具；Vivado HLS 2019.2 需要在独立环境和独立输出目录复跑。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _aggregate_output(
    output: Path,
    manifest: MutableMapping[str, Any],
    float_labels: np.ndarray,
    float_predictions: np.ndarray,
    float_logits: np.ndarray,
    float_accuracy: float,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = _finish_records(
        records,
        float_accuracy,
        float_labels,
        float_predictions,
        float_logits,
        str(manifest.get("mode", "full")),
    )
    for record in records:
        _write_variant_state(output / f"w{int(record['width'])}", record)
        manifest["variants"][str(record["width"])] = {
            "directory": str(output / f"w{int(record['width'])}"),
            "metrics": str(output / f"w{int(record['width'])}" / "metrics.json"),
            "csim_status": record.get("csim_status"),
            "synth_status": record.get("synth_status"),
            "parse_status": record.get("parse_status"),
        }
    _write_sweep_csv(output / "precision_sweep.csv", records)
    _json_write(output / "summary.json", summary)
    _write_plots(output, records, float_accuracy)
    _write_report_markdown(output, manifest, summary)
    _json_write(output / "manifest.json", manifest)
    return summary


def _run_experiment(args: argparse.Namespace) -> int:
    blob_path = Path(args.blob).expanduser().resolve()
    if not blob_path.is_file():
        raise ValidationError(f"blob does not exist: {blob_path}")
    metadata_path = (
        Path(args.blob_metadata).expanduser().resolve()
        if args.blob_metadata
        else blob_path.with_suffix(".metadata.json")
    )
    if not metadata_path.is_file():
        raise ValidationError(f"blob metadata does not exist: {metadata_path}")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ValidationError(f"output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValidationError(f"output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)

    mode = "python-only" if args.python_only else "full"
    executable = None
    if mode == "full":
        if not args.hls_executable:
            raise ValidationError("full run requires --hls-executable")
        executable = _tool_path(args.hls_executable)

    metadata = _validate_blob_metadata(blob_path, metadata_path)
    blob = read_lenet_blob(blob_path, EXPECTED_MNIST_COUNT)
    float_logits, float_predictions = predict_images(blob.images, blob.parameters)
    float_accuracy = float(_accuracy(float_predictions, blob.labels))
    write_result_csv(output / "float.csv", blob.labels, float_predictions, float_logits)
    float_ranges = collect_float_ranges(blob.images, blob.parameters)
    _json_write(output / "ranges.json", {"float_reference": float_ranges, "python_approx": {}})

    manifest = _make_manifest(
        blob_path, metadata_path, metadata, output, mode, executable
    )
    if executable:
        manifest["toolchain"].update(_probe_tool(executable))
        manifest["toolchain"]["status"] = "used"
    _json_write(output / "manifest.json", manifest)

    records: List[Dict[str, Any]] = []
    # The order is intentionally fixed: establish the 16-bit baseline first.
    for width in (16, *range(8, 16)):
        variant = output / f"w{width}"
        variant.mkdir()
        record = _base_record(width)
        approx_logits, approx_ranges = python_approx_forward(
            blob.images, blob.parameters, width
        )
        approx_predictions = approx_logits.argmax(axis=1).astype(np.int32)
        write_result_csv(
            variant / "python_approx.csv",
            blob.labels,
            approx_predictions,
            approx_logits,
        )
        ranges = _json_read(output / "ranges.json")
        ranges["python_approx"][str(width)] = approx_ranges
        _json_write(output / "ranges.json", ranges)
        record["python_approx_accuracy_pct"] = _accuracy(approx_predictions, blob.labels)
        record["python_approx_agreement_pct"] = 100.0 * float(
            np.count_nonzero(approx_predictions == float_predictions)
        ) / EXPECTED_MNIST_COUNT
        record["float_accuracy_pct"] = float_accuracy
        if mode == "python-only":
            record["csim_status"] = "skipped"
            record["synth_status"] = "skipped"
            record["parse_status"] = "skipped"
        else:
            record.update(_run_hls_variant(width, blob_path, variant, executable, blob.labels, manifest))
            if record.get("csim_status") == "ok":
                try:
                    hls_predictions, hls_logits = _read_result_arrays(
                        variant / "hls.csv", blob.labels, EXPECTED_MNIST_COUNT
                    )
                    record["hls_accuracy_pct"] = _accuracy(hls_predictions, blob.labels)
                    record["agreement_pct"] = 100.0 * float(
                        np.count_nonzero(hls_predictions == float_predictions)
                    ) / EXPECTED_MNIST_COUNT
                    differences = np.abs(hls_logits - float_logits)
                    record["logit_mae"] = float(differences.mean())
                    record["logit_max_abs_error"] = float(differences.max())
                except (OSError, ValidationError) as exc:
                    record["csim_status"] = "failed"
                    record["error"] = f"HLS result validation failed: {exc}"
        records.append(record)
        _write_variant_state(variant, record)
        manifest["variants"][str(width)] = {
            "directory": str(variant),
            "metrics": str(variant / "metrics.json"),
            "csim_status": record["csim_status"],
            "synth_status": record["synth_status"],
            "parse_status": record["parse_status"],
        }
        _json_write(output / "manifest.json", manifest)

    _aggregate_output(
        output,
        manifest,
        blob.labels,
        float_predictions,
        float_logits,
        float_accuracy,
        records,
    )
    print(json.dumps(_json_read(output / "summary.json"), ensure_ascii=False, indent=2))
    return 0 if mode == "python-only" else (0 if _json_read(output / "summary.json")["status"] == "COMPLETE" else 1)


def _load_offline_run(output: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest = _json_read(output / "manifest.json")
    if manifest.get("scan", {}).get("widths") != list(WIDTHS):
        raise ValidationError("manifest does not describe the required 8..16 scan")
    blob_info = manifest.get("blob", {})
    expected_count = int(blob_info.get("sample_count", EXPECTED_MNIST_COUNT))
    if expected_count != EXPECTED_MNIST_COUNT:
        raise ValidationError("offline report requires sample_count=10000")
    float_rows = read_result_csv(output / "float.csv")
    if len(float_rows) != EXPECTED_MNIST_COUNT:
        raise ValidationError("float.csv is not a complete 10,000-sample result")
    float_labels = np.asarray([int(row["expected"]) for row in float_rows], dtype=np.int32)
    float_predictions = np.asarray([int(row["prediction"]) for row in float_rows], dtype=np.int32)
    float_logits = np.asarray([row["logits"] for row in float_rows], dtype=np.float64)
    records = []
    for width in (16, *range(8, 16)):
        variant = output / f"w{width}"
        metrics_path = variant / "metrics.json"
        record = _base_record(width)
        if metrics_path.is_file():
            stored = _json_read(metrics_path)
            record.update(stored)
        try:
            approx_predictions, approx_logits = _read_result_arrays(
                variant / "python_approx.csv", float_labels, EXPECTED_MNIST_COUNT
            )
            record["python_approx_accuracy_pct"] = _accuracy(approx_predictions, float_labels)
            record["python_approx_agreement_pct"] = 100.0 * float(
                np.count_nonzero(approx_predictions == float_predictions)
            ) / EXPECTED_MNIST_COUNT
        except (OSError, ValidationError) as exc:
            record["error"] = f"Python result invalid: {exc}"
            record["csim_status"] = "failed" if manifest.get("mode") == "full" else "skipped"
        hls_path = variant / "hls.csv"
        if hls_path.is_file():
            try:
                hls_predictions, hls_logits = _read_result_arrays(
                    hls_path, float_labels, EXPECTED_MNIST_COUNT
                )
                record["hls_accuracy_pct"] = _accuracy(hls_predictions, float_labels)
                record["agreement_pct"] = 100.0 * float(
                    np.count_nonzero(hls_predictions == float_predictions)
                ) / EXPECTED_MNIST_COUNT
                difference = np.abs(hls_logits - float_logits)
                record["logit_mae"] = float(difference.mean())
                record["logit_max_abs_error"] = float(difference.max())
                record["csim_status"] = "ok"
            except (OSError, ValidationError) as exc:
                record["csim_status"] = "failed"
                record["error"] = (record.get("error") + "; " if record.get("error") else "") + str(exc)
        report_path = variant / "csynth_report.xml"
        if not report_path.is_file():
            report_path = variant / "csynth_report.rpt"
        if report_path.is_file():
            record.update(parse_csynth_report(report_path))
            record["parse_status"] = record.get("parse_status", "failed")
        elif manifest.get("mode") == "full":
            record["parse_status"] = "failed"
            record["synth_status"] = "failed" if record["csim_status"] == "ok" else "skipped"
        records.append(record)
    summary = _finish_records(
        records,
        100.0 * float(np.count_nonzero(float_predictions == float_labels)) / EXPECTED_MNIST_COUNT,
        float_labels,
        float_predictions,
        float_logits,
        str(manifest.get("mode", "full")),
    )
    return manifest, {"summary": summary, "labels": float_labels, "predictions": float_predictions, "logits": float_logits, "records": records}


def _report_experiment(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    if not output.is_dir():
        raise ValidationError(f"experiment output does not exist: {output}")
    manifest, data = _load_offline_run(output)
    summary = data["summary"]
    for record in data["records"]:
        _write_variant_state(output / f"w{int(record['width'])}", record)
    _write_sweep_csv(output / "precision_sweep.csv", data["records"])
    _json_write(output / "summary.json", summary)
    _write_plots(output, data["records"], summary["float_accuracy_pct"])
    _write_report_markdown(output, manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the 8..16 precision sweep")
    run.add_argument("--blob", required=True, help="complete 10,000-sample LeNet blob")
    run.add_argument("--blob-metadata", help="metadata generated by make-blob")
    run.add_argument("--hls-executable", help="vivado_hls or vitis_hls executable")
    run.add_argument("--python-only", action="store_true", help="skip HLS explicitly")
    run.add_argument("--output", required=True, help="new or empty experiment directory")
    run.set_defaults(function=_run_experiment)

    report = subparsers.add_parser("report", help="rebuild a report without running HLS")
    report.add_argument("--output", required=True, help="existing experiment directory")
    report.set_defaults(function=_report_experiment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (OSError, ValidationError, ValueError) as exc:
        print(f"precision experiment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
