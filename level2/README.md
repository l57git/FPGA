# LeNet Level 2: real-scene preprocessing

This module converts photographed handwritten digits to the `28x28` single-channel MNIST convention and evaluates them with the Level 1 model.

## Reproducible stress test

```bat
python tools\level2_validation.py benchmark --images ..\level1\data\mnist\t10k-images-idx3-ubyte --labels ..\level1\data\mnist\t10k-labels-idx1-ubyte --parameters ..\level1\data\lenet_accuracy_1.bin --output results\stress_test --count 1000
```

Compare the resulting Python predictions with Vivado HLS fixed-point C simulation:

```bat
run_stress_hls_windows.bat
```

## Self-collected photos

Place photos in `data/self_collected/`. Each filename must start with the correct label and an underscore, for example `7_01.jpg`.

```bat
python tools\level2_validation.py evaluate-folder --input data\self_collected --parameters ..\level1\data\lenet_accuracy_1.bin --output results\self_collected_predictions.csv
```

The stress-test results are development evidence and must not be described as self-collected data. Formal Level 2 results require photos taken by the group.

The generated report directory contains raw per-sample CSV files, the Vivado HLS CSim log, preprocessing examples, confusion matrices, and a presentation-ready result overview.
