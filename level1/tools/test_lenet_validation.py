import argparse
import csv
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lenet_validation import (
    BATCH_MAGIC,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MNIST_IMAGE_MAGIC,
    MNIST_LABEL_MAGIC,
    PARAMETER_SPECS,
    ValidationError,
    command_compare,
    compare_result_files,
    conv2d_valid,
    lenet_forward,
    max_pool_2x2,
    predict_images,
    read_lenet_blob,
    read_mnist_test_set,
    read_result_csv,
    write_lenet_blob,
    write_result_csv,
)


class LeNetValidationTests(unittest.TestCase):
    def test_tracked_blob_and_float_prediction(self):
        blob = read_lenet_blob(
            Path(__file__).parents[1] / "data" / "lenet_accuracy_1.bin",
            expected_count=1,
        )
        logits, predictions = predict_images(blob.images, blob.parameters, batch_size=1)
        self.assertEqual(blob.labels.tolist(), [7])
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(predictions.tolist(), [7])

    def test_blob_uses_interleaved_sample_records(self):
        source = Path(__file__).parents[1] / "data" / "lenet_accuracy_1.bin"
        parameters = read_lenet_blob(source, expected_count=1).parameters
        labels = np.array([7, 3], dtype=np.int32)
        images = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        images[0, 0, 0] = 0.25
        images[1, 0, 0] = 0.75
        with tempfile.TemporaryDirectory() as directory:
            blob_path = Path(directory) / "two_samples.bin"
            write_lenet_blob(blob_path, parameters, labels, images)
            decoded = read_lenet_blob(blob_path, expected_count=2)
            np.testing.assert_array_equal(decoded.labels, labels)
            np.testing.assert_array_equal(decoded.images, images)

    def test_forward_shapes_and_flattened_layout(self):
        parameters = {
            name: np.zeros(shape, dtype=np.float32)
            for name, _, shape in PARAMETER_SPECS
        }
        parameters["fc3_b"][3] = 1.0
        images = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        inputs = images[:, None, :, :]
        conv1 = conv2d_valid(
            inputs, parameters["conv1_w"][:, None, :, :], parameters["conv1_b"]
        )
        pool1 = max_pool_2x2(np.maximum(conv1, 0.0))
        conv2 = conv2d_valid(pool1, parameters["conv2_w"], parameters["conv2_b"])
        pool2 = max_pool_2x2(np.maximum(conv2, 0.0))
        self.assertEqual(conv1.shape, (2, 6, 24, 24))
        self.assertEqual(pool1.shape, (2, 6, 12, 12))
        self.assertEqual(conv2.shape, (2, 16, 8, 8))
        self.assertEqual(pool2.shape, (2, 16, 4, 4))
        logits = lenet_forward(images, parameters)
        self.assertEqual(logits.shape, (2, 10))
        np.testing.assert_array_equal(logits[:, 3], [1.0, 1.0])
        np.testing.assert_array_equal(np.argmax(logits, axis=1), [3, 3])

    def test_idx_validation_and_corrupt_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_path = root / "t10k-images-idx3-ubyte"
            labels_path = root / "t10k-labels-idx1-ubyte"
            images = np.zeros((10000, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            labels = (np.arange(10000, dtype=np.int32) % 10).astype(np.uint8)
            with images_path.open("wb") as stream:
                stream.write(
                    struct.pack(">IIII", MNIST_IMAGE_MAGIC, 10000, IMAGE_HEIGHT, IMAGE_WIDTH)
                )
                stream.write(images.tobytes())
            with labels_path.open("wb") as stream:
                stream.write(struct.pack(">II", MNIST_LABEL_MAGIC, 10000))
                stream.write(labels.tobytes())

            dataset = read_mnist_test_set(images_path, labels_path)
            self.assertEqual(dataset.images.shape, (10000, 28, 28))
            self.assertEqual(dataset.labels.shape, (10000,))
            self.assertEqual(dataset.labels[-1], 9)

            images_path.write_bytes(images_path.read_bytes()[:-1])
            with self.assertRaises(ValidationError):
                read_mnist_test_set(images_path, labels_path)

    def test_blob_rejects_trailing_data(self):
        source = Path(__file__).parents[1] / "data" / "lenet_accuracy_1.bin"
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.bin"
            broken.write_bytes(source.read_bytes() + b"extra")
            with self.assertRaises(ValidationError):
                read_lenet_blob(broken)

    def test_compare_success_and_threshold_failure(self):
        labels = np.array([1, 2], dtype=np.int32)
        float_predictions = np.array([1, 2], dtype=np.int32)
        hls_predictions = np.array([1, 3], dtype=np.int32)
        logits = np.zeros((2, 10), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            float_path = root / "float.csv"
            hls_path = root / "hls.csv"
            report_path = root / "report.json"
            mismatch_path = root / "mismatches.csv"
            write_result_csv(float_path, labels, float_predictions, logits)
            write_result_csv(hls_path, labels, hls_predictions, logits)
            report = compare_result_files(
                float_path, hls_path, report_path, mismatch_path, threshold=90.0
            )
            self.assertEqual(report["float_accuracy_percent"], 100.0)
            self.assertEqual(report["hls_accuracy_percent"], 50.0)
            self.assertEqual(report["prediction_consistency_percent"], 50.0)
            self.assertFalse(report["hls_threshold_passed"])
            self.assertEqual(len(read_result_csv(float_path)), 2)
            with mismatch_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["index"], "1")

            lines = hls_path.read_text(encoding="utf-8").splitlines()
            hls_path.write_text(
                "\n".join([lines[0], lines[2], lines[1]]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                read_result_csv(hls_path)

            hls_path.write_text(
                "index,expected,prediction,logit_0,logit_1,logit_2,logit_3,logit_4,logit_5,logit_6,logit_7,logit_8,logit_9\n"
                "0,1,1,0,0,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                compare_result_files(
                    float_path, hls_path, report_path, mismatch_path, threshold=90.0
                )

            write_result_csv(hls_path, labels, hls_predictions, logits)
            command_args = argparse.Namespace(
                float_results=float_path,
                hls_results=hls_path,
                report=report_path,
                mismatches=mismatch_path,
                threshold=90.0,
            )
            self.assertEqual(command_compare(command_args), 1)
            self.assertTrue(report_path.exists())
            write_result_csv(hls_path, labels, float_predictions, logits)
            self.assertEqual(command_compare(command_args), 0)


if __name__ == "__main__":
    unittest.main()
