# LeNet Level 1

Vivado HLS 2019.2 implementation of LeNet inference for MNIST.

## Network

`Conv1 + bias + ReLU -> MaxPool -> Conv2 + bias + ReLU -> MaxPool -> FC1 + bias + ReLU -> FC2 + bias + ReLU -> FC3 + bias -> Argmax`

The hardware inference path intentionally omits softmax because softmax does not change the argmax classification result.

## Environment

- Vivado HLS 2019.2
- Target: `xc7z020clg400-1`
- Clock: 10 ns

## Run C simulation

Open a Vivado HLS Command Prompt in this directory and run:

```bat
set LENET_ACCURACY_BLOB=%CD%\data\lenet_accuracy_1.bin
vivado_hls -f run_hls.tcl
```

The included blob contains one smoke-test sample. Its verified result is label 7, prediction 7.

## Complete MNIST validation

The validation tools use the official MNIST test split and normalize pixels with `uint8 / 255.0`. Download and unpack these files into `data/mnist/`:

```sh
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -o data/mnist/t10k-images-idx3-ubyte.gz
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -o data/mnist/t10k-labels-idx1-ubyte.gz
gzip -df data/mnist/t10k-images-idx3-ubyte.gz data/mnist/t10k-labels-idx1-ubyte.gz
```

Validate the inputs, generate the HLS batch blob, and run the NumPy reference model:

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

Run HLS C simulation with the optional result CSV enabled, using the HLS environment described in the repository `CLAUDE.md`:

```sh
mkdir -p results
LENET_SKIP_SYNTH=1 \
LENET_ACCURACY_BLOB="$PWD/data/lenet_accuracy_10000.bin" \
LENET_RESULT_CSV="$PWD/results/hls_results.csv" \
vivado_hls -f run_hls.tcl
```

Compare both result files and apply the Level 1 fixed-point threshold:

```sh
python3 tools/lenet_validation.py compare \
  --float-results results/float_results.csv \
  --hls-results results/hls_results.csv \
  --report results/validation_report.json \
  --mismatches results/mismatches.csv \
  --threshold 90
```

The complete IDX files, batch blob, and per-sample CSV files are reproducible inputs or outputs and are ignored by Git. The JSON summaries, mismatch CSV, and validation report can be retained as experiment evidence.

