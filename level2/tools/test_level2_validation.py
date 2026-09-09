import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level2_validation import preprocess_image, raw_resize, scan_labeled_dataset


class Level2PreprocessingTests(unittest.TestCase):
    def test_blank_background_returns_zero_image(self):
        result = preprocess_image(Image.new("L", (80, 60), 240))
        self.assertEqual(result.shape, (28, 28))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result.max()), 0.0)

    def test_dark_digit_on_light_background_is_cropped_and_centered(self):
        image = Image.new("L", (100, 70), 245)
        draw = ImageDraw.Draw(image)
        draw.line((72, 10, 72, 58), fill=10, width=8)
        result = preprocess_image(image)
        yy, xx = np.indices(result.shape)
        mass = float(result.sum())
        self.assertGreater(mass, 5.0)
        self.assertAlmostEqual(float((yy * result).sum() / mass), 13.5, delta=1.0)
        self.assertAlmostEqual(float((xx * result).sum() / mass), 13.5, delta=1.0)

    def test_light_digit_on_dark_background_has_mnist_polarity(self):
        image = Image.new("L", (60, 60), 5)
        ImageDraw.Draw(image).ellipse((20, 8, 42, 52), outline=240, width=6)
        result = preprocess_image(image)
        self.assertGreater(float(result.max()), 0.9)
        self.assertLess(float(result[0, 0]), 0.05)

    def test_raw_resize_has_expected_range(self):
        image = Image.new("L", (40, 40), 255)
        ImageDraw.Draw(image).line((5, 5, 35, 35), fill=0, width=4)
        result = raw_resize(image)
        self.assertEqual(result.shape, (28, 28))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_scan_class_directories_and_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in (0, 1):
                folder = root / str(label)
                folder.mkdir()
                Image.new("L", (12, 10), label * 100).save(folder / f"digit{label}_sample.png")
            (root / "1" / ".DS_Store").write_bytes(b"metadata")
            (root / "1" / "notes.txt").write_text("ignored", encoding="utf-8")

            samples, excluded = scan_labeled_dataset(root)

            self.assertEqual([sample["label"] for sample in samples], [0, 1])
            self.assertEqual([sample["width"] for sample in samples], [12, 12])
            self.assertEqual(len(excluded), 2)
            self.assertEqual(
                {item["reason"] for item in excluded},
                {"metadata_or_hidden", "unsupported_extension"},
            )

    def test_scan_rejects_filename_directory_label_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "2"
            root.mkdir(parents=True)
            Image.new("L", (12, 10), 0).save(root / "digit3_sample.png")

            with self.assertRaisesRegex(ValueError, "conflicts"):
                scan_labeled_dataset(root.parent)


if __name__ == "__main__":
    unittest.main()
