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

For class-directory data such as `data/data/raw/0` through `9`, run the reproducible dataset flow from the repository root:

```sh
python3 level2/tools/level2_validation.py evaluate-dataset \
  --input data/data/raw \
  --parameters level1/data/lenet_accuracy_1.bin \
  --output level2/results/self_collected
```

This writes a manifest, Python predictions, confusion matrices, a presentation overview, and `preprocessed_self_collected.bin` for the HLS testbench. Run `run_self_collected_hls.sh` (or the Windows `.bat` equivalent) in an installed Vivado HLS environment to perform C Simulation and compare the HLS predictions with `python_results.csv`. The HLS script intentionally runs C Simulation only; synthesis remains a separate project-level check.

The stress-test results are development evidence and must not be described as self-collected data. Formal Level 2 results require photos taken by the group.

The generated report directory contains raw per-sample CSV files, the Python/HLS comparison, preprocessing examples, confusion matrices, and a presentation-ready result overview. The detailed Vivado/Vitis HLS CSim log is retained in the ignored `level2/hls_work/` workspace.
