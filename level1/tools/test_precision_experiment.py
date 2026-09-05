import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lenet_validation import PARAMETER_SPECS, ValidationError, write_result_csv
from precision_experiment import (
    ACCURACY_THRESHOLD,
    MAX_LOSS_PP,
    WIDTHS,
    _base_record,
    _finish_records,
    _read_result_arrays,
    _write_report_markdown,
    parse_csynth_report,
    python_approx_forward,
    quantize,
)


class PrecisionExperimentTests(unittest.TestCase):
    def test_rounding_and_saturation_boundaries(self):
        values = quantize(np.array([0.125, -0.125, 32.0, -33.0]), 8)
        np.testing.assert_array_equal(values, [0.25, 0.0, 31.75, -32.0])

    def test_invalid_width_is_rejected(self):
        with self.assertRaises(ValidationError):
            quantize(np.array([0.0]), 7)
        with self.assertRaises(ValidationError):
            quantize(np.array([0.0]), 17)

    def test_approximation_keeps_argmax_tie_at_lowest_index(self):
        parameters = {
            name: np.zeros(shape, dtype=np.float32)
            for name, _, shape in PARAMETER_SPECS
        }
        parameters["fc3_b"][3] = 1.0
        parameters["fc3_b"][5] = 1.0
        images = np.zeros((1, 28, 28), dtype=np.float32)
        logits, _ = python_approx_forward(images, parameters, 8, batch_size=1)
        self.assertEqual(int(logits.argmax(axis=1)[0]), 3)

    def test_result_validation_rejects_nan_and_misaligned_labels(self):
        labels = np.array([1, 2], dtype=np.int32)
        predictions = np.array([1, 2], dtype=np.int32)
        logits = np.zeros((2, 10), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.csv"
            write_result_csv(valid, labels, predictions, logits)
            parsed_predictions, parsed_logits = _read_result_arrays(valid, labels, 2)
            np.testing.assert_array_equal(parsed_predictions, predictions)
            np.testing.assert_array_equal(parsed_logits, logits)

            broken = root / "broken.csv"
            broken.write_text(
                valid.read_text(encoding="utf-8").replace(
                    "0,1,1,", "1,1,1,"
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                _read_result_arrays(broken, labels, 2)

            nan_file = root / "nan.csv"
            nan_file.write_text(
                valid.read_text(encoding="utf-8").replace(",0,", ",nan,", 1),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                _read_result_arrays(nan_file, labels, 2)

    def test_selection_boundary_uses_unrounded_half_point(self):
        labels = np.zeros(1, dtype=np.int32)
        predictions = np.zeros(1, dtype=np.int32)
        logits = np.zeros((1, 10), dtype=np.float64)

        def records(width8_accuracy):
            result = []
            for width in WIDTHS:
                record = _base_record(width)
                record.update(
                    {
                        "csim_status": "ok",
                        "synth_status": "ok",
                        "parse_status": "ok",
                        "hls_accuracy_pct": 98.38,
                    }
                )
                if width == 8:
                    record["hls_accuracy_pct"] = width8_accuracy
                if width == 9:
                    record["hls_accuracy_pct"] = 97.88
                result.append(record)
            return result

        summary = _finish_records(
            records(97.88), 98.37, labels, predictions, logits, "full"
        )
        self.assertEqual(summary["recommendation_width"], 8)
        self.assertLessEqual(MAX_LOSS_PP, 0.5)

        summary = _finish_records(
            records(97.87), 98.37, labels, predictions, logits, "full"
        )
        self.assertEqual(summary["recommendation_width"], 9)
        self.assertEqual(summary["accuracy_threshold_pct"], ACCURACY_THRESHOLD)

    def test_csynth_xml_metrics_are_parsed_without_zero_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lenet_accel_csynth.xml"
            path.write_text(
                "<profile><AreaEstimates><Resources>"
                "<LUT>18024</LUT><FF>12815</FF><DSP48E>13</DSP48E><BRAM_18K>14</BRAM_18K>"
                "</Resources></AreaEstimates><PerformanceEstimates>"
                "<SummaryOfOverallLatency><LatencyWorst>484541</LatencyWorst></SummaryOfOverallLatency>"
                "<SummaryOfTimingAnalysis><EstimatedClockPeriod>7.3</EstimatedClockPeriod></SummaryOfTimingAnalysis>"
                "</PerformanceEstimates></profile>",
                encoding="utf-8",
            )
            metrics = parse_csynth_report(path)
            self.assertEqual(metrics["parse_status"], "ok")
            self.assertEqual(metrics["lut"], 18024)
            self.assertEqual(metrics["latency_ns_target"], 4845410.0)

    def test_missing_csynth_metric_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.xml"
            path.write_text(
                "<profile><AreaEstimates><Resources>"
                "<LUT>1</LUT><FF>2</FF><DSP48E>3</DSP48E><BRAM_18K>4</BRAM_18K>"
                "</Resources></AreaEstimates></profile>",
                encoding="utf-8",
            )
            metrics = parse_csynth_report(path)
            self.assertEqual(metrics["parse_status"], "failed")
            self.assertIsNone(metrics["latency_cycles_max"])
            self.assertIsNone(metrics["estimated_clock_ns"])

    def test_zero_resource_baseline_is_reported_as_not_computable(self):
        variants = []
        for width in (16, *range(8, 16)):
            record = _base_record(width)
            record.update(
                {"lut": 0 if width == 16 else 1, "ff": 2, "dsp": 3, "bram_18k": 4}
            )
            variants.append(record)
        summary = {
            "status": "INCOMPLETE",
            "mode": "python-only",
            "sample_count": 10000,
            "float_accuracy_pct": 98.37,
            "baseline_hls_accuracy_pct": None,
            "recommendation_width": None,
            "recommendation_reason": "incomplete",
            "variants": variants,
        }
        manifest = {
            "toolchain": {},
            "blob": {"sha256": "blob", "parameter_region_sha256": "params"},
            "git": {"revision": "test", "dirty": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_report_markdown(output, manifest, summary)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("不可计算", report)

    def test_stage_failure_cannot_produce_a_recommendation(self):
        labels = np.zeros(1, dtype=np.int32)
        predictions = np.zeros(1, dtype=np.int32)
        logits = np.zeros((1, 10), dtype=np.float64)
        records = []
        for width in (16, *range(8, 16)):
            record = _base_record(width)
            record.update(
                {
                    "csim_status": "ok",
                    "synth_status": "ok",
                    "parse_status": "ok",
                    "hls_accuracy_pct": 98.38,
                    "lut": 18020,
                }
            )
            records.append(record)
        failed = next(record for record in records if int(record["width"]) == 10)
        failed["synth_status"] = "failed"
        failed["lut"] = 12345
        summary = _finish_records(records, 98.37, labels, predictions, logits, "full")
        self.assertIsNone(summary["recommendation_width"])
        self.assertEqual(failed["lut"], 12345)


if __name__ == "__main__":
    unittest.main()
