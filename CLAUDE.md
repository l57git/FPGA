# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

- This repository contains the selected **Task 2 / LeNet** implementation. The course requirements are summarized in `docs/任务要求与LeNet路线核查.md`; the original task PDF is intentionally not tracked.
- The current code is the Level 1 hardware-inference baseline for MNIST. It is intended for Vivado HLS C simulation and synthesis, not model training or software deployment.
- The expected toolchain is Vivado HLS 2019.2, targeting `xc7z020clg400-1` with a 10 ns clock. The current shell does not provide a `vivado_hls` executable, so HLS commands must be run from an installed Vivado HLS 2019.2 command environment.
- The repository includes the one-sample regression blob, reproducible Python validation tools, and Level 1 implementation. Complete MNIST inputs and per-sample results are generated under Git-ignored paths; the tracked experiment summaries record the current validation evidence.

## Architecture

`level1/src/lenet.hpp` defines the network dimensions, fixed-point types, flattened parameter counts, and the public top-level function `lenet_accel`. `level1/src/lenet.cpp` implements the complete inference pipeline:

```text
28x28 image
  -> Conv1 (6 output channels, 5x5) + bias + ReLU
  -> 2x2 max pool
  -> Conv2 (16 output channels, 5x5 over 6 channels) + bias + ReLU
  -> 2x2 max pool
  -> FC1 (256 -> 120) + bias + ReLU
  -> FC2 (120 -> 84) + bias + ReLU
  -> FC3 (84 -> 10) + bias
  -> logits and argmax prediction
```

- All tensors and parameters use `data_t = ap_fixed<16, 6, AP_RND, AP_SAT>`; accumulators use `acc_t = ap_fixed<32, 14, AP_RND, AP_SAT>`.
- Parameters are passed as flattened arrays. Conv1 weights are indexed `[output][ky][kx]`, Conv2 weights `[output][input][ky][kx]`, and fully connected weights `[output][input]`. The FC input is the flattened `[16][4][4]` second pooling result in channel-major order.
- Intermediate feature maps are function-local static arrays. The HLS interface maps the image and logits to `gmem0`, all weights and biases to `gmem1`, and exposes all arguments through the `control` AXI-Lite bundle.
- Softmax is intentionally absent because only the class argmax is required; changing this would add work without changing the prediction.

## Testbench and data format

`level1/tb/tb_lenet.cpp` is the only testbench and has three invocation modes:

- With no argument, it zeroes the image and all parameters, sets class-3 FC3 bias to `1.0`, and checks the deterministic class-3 smoke test.
- With one argument, it reads a binary batch file. The file starts with two little-endian `int32_t` values: magic `-20260902` and a positive batch count. It then stores all parameter arrays in this order: `conv1_w`, `conv1_b`, `conv2_w`, `conv2_b`, `fc1_w`, `fc1_b`, `fc2_w`, `fc2_b`, `fc3_w`, `fc3_b`. Each sample consists of an `int32_t` expected label followed by 784 little-endian `float` pixels. Floats are converted into `data_t` before inference.
- With two arguments, the first is the same batch file and the second is an optional CSV result path. The CSV contains `index`, `expected`, `prediction`, and ten logits per sample.
- The batch mode returns success only when fixed-point accuracy is at least 90%, prints progress at least every 100 samples, shows logits for the first five samples, and reports up to 20 misclassifications.
- `level1/data/lenet_accuracy_1.bin` is the tracked one-sample blob; its documented expected result is label 7 and prediction 7.

If dimensions, tensor layout, or parameter ordering change, update the constants and interface depths in `lenet.hpp`/`lenet.cpp` together with the testbench's blob ordering and any generated data. Keep the implementation within the C++ subset supported by Vivado HLS 2019.2.

## HLS commands

Run commands from `level1/` in a Vivado HLS 2019.2 command prompt. `run_hls.tcl` resolves paths relative to itself, creates the ignored `level1/hls_work/` workspace, resets the project and solution, adds the source and testbench, sets the top function, part, and clock, then runs C simulation and synthesis.

Windows batch prompt:

```bat
cd level1
set LENET_ACCURACY_BLOB=%CD%\data\lenet_accuracy_1.bin
vivado_hls -f run_hls.tcl
```

POSIX shell:

```sh
cd level1
LENET_ACCURACY_BLOB="$PWD/data/lenet_accuracy_1.bin" vivado_hls -f run_hls.tcl
```

Useful focused runs:

```sh
# C simulation only, using the tracked one-sample batch
LENET_SKIP_SYNTH=1 LENET_ACCURACY_BLOB="$PWD/data/lenet_accuracy_1.bin" vivado_hls -f run_hls.tcl

# Synthesis only
LENET_SKIP_CSIM=1 vivado_hls -f run_hls.tcl

# No blob: run the built-in class-3 smoke test, without synthesis
LENET_SKIP_SYNTH=1 vivado_hls -f run_hls.tcl
```

The Tcl script accepts `LENET_SKIP_CSIM=1`, `LENET_SKIP_SYNTH=1`, and the optional `LENET_RESULT_CSV` output path. If `LENET_ACCURACY_BLOB` is unset, it looks for `data/lenet_accuracy.bin`; that file is not currently tracked, so the testbench falls back to the no-argument smoke test. There is no Makefile, CMake project, package manager, standalone lint command, or separate unit-test framework; Vivado HLS C simulation is the hardware validation command.

## Complete MNIST and Python comparison

`level1/tools/lenet_validation.py` provides the reproducible experiment flow. It validates the official IDX test split, parses the existing parameter blob, generates the 10,000-sample HLS blob, runs a NumPy float32 reference, validates result CSVs, and compares floating-point and fixed-point predictions. MNIST pixels are normalized as `uint8 / 255.0`; the Python and HLS inputs use the same parameter source and sample order.

Run the Python checks from `level1/`:

```sh
python3 tools/lenet_validation.py validate-mnist \
  --images data/mnist/t10k-images-idx3-ubyte \
  --labels data/mnist/t10k-labels-idx1-ubyte
python3 tools/lenet_validation.py make-blob \
  --parameters data/lenet_accuracy_1.bin \
  --images data/mnist/t10k-images-idx3-ubyte \
  --labels data/mnist/t10k-labels-idx1-ubyte \
  --output data/lenet_accuracy_10000.bin
python3 tools/lenet_validation.py run-float \
  --blob data/lenet_accuracy_10000.bin \
  --results results/float_results.csv \
  --summary results/float_summary.json \
  --expected-count 10000
```

For HLS CSV export, set `LENET_RESULT_CSV` alongside `LENET_ACCURACY_BLOB`, then compare with the `compare` subcommand. The large MNIST files, complete blob, and per-sample CSV outputs are ignored by Git; summaries, mismatch lists, and reports can be retained.

Generated HLS files belong under `level1/hls_work/` and are ignored by Git. To discard only that generated workspace before a clean rerun:

```sh
rm -rf level1/hls_work
```
