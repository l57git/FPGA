import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level2_validation import preprocess_image, raw_resize


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


if __name__ == "__main__":
    unittest.main()
