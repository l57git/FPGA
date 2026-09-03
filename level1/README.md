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

The included blob contains one smoke-test sample. Its verified result is label 7, prediction 7. A full MNIST accuracy blob and synthesis/cosimulation reports still need to be added.

