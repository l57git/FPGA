# 真实图片预处理 v1

本次保留原有 `tools/level2_validation.py` 和 `results/self_collected/`，新增可独立调用的预处理器，避免把历史 HLS 验证结果误用于新输入。

## 已完成的实测结果

|方法|全部 503 张|开发集 358 张|图像级留出测试 145 张|
|---|---:|---:|---:|
|直接缩放|112 / 503，22.27%|79 / 358，22.07%|33 / 145，22.76%|
|旧预处理|56 / 503，11.13%|37 / 358，10.34%|19 / 145，13.10%|
|新预处理|429 / 503，85.29%|303 / 358，84.64%|126 / 145，86.90%|

8 张拒识计为错误，没有从准确率分母中删除。模型参数不变。Python W16 整数参考也为 429 / 503，但浮点/整数类别一致为 502 / 503，并非全部一致。尚未运行新 HLS C Simulation。

这是内部图像级划分，不是跨书写者、跨拍摄批次或外部独立盲测。原项目以前已评估过全体 503 张。无法从文件名证明真实采集来源。新版本没有按文件名、标签、类别或模型输出切换预处理规则。

## 文件入口

- `tools/preprocess_robust.py`：纯像素算法；默认不加粗。
- `tools/preprocess_photo.py`：处理一张不带标签的新照片。
- `tools/preprocessing_experiment.py`：数据审计、固定划分与旧结果复现。
- `tools/evaluate_robust.py`：开发集消融、配置冻结、留出评估和 HLS 数据导出。
- `tools/report_robust.py`：只读实验结果，制作完整审查页面与报告。
- `results/preprocessing_v1/改进实验报告.md`：实验条件、结果、限制。
- `results/preprocessing_v1/完整样本审查.html`：全部 503 张原图缩略图、旧/新输入与预测；可筛选失败样本。
- `results/preprocessing_v1/all_test_failures.jpg`：全部 19 张测试集失败，没有挑选。

## 环境

当前在本机 `C:/Users/32360/miniconda3/python.exe` 运行。依赖已存在，没有更改全局环境。版本见结果目录 `environment.json`。其他机器可在独立虚拟环境安装 `requirements-preprocessing.txt`。

## 新照片

在仓库根目录运行：

```powershell
python level2/tools/preprocess_photo.py "你的照片.png" --output "level2/results/new_photo_01"
```

输出 28×28 PNG、float32 NPY 和诊断 JSON。输出目录必须不存在，避免覆盖。状态非 `ok` 时返回码为 2；这是拒识，应交给用户重新拍摄或人工复核，不应把全零图强行识别为数字。

代码调用：

```python
from PIL import Image
from preprocess_robust import preprocess
with Image.open(photo_path) as photo:
    image28, diagnostics = preprocess(photo)
if diagnostics['status'] == 'ok':
    # image28: float32 (28,28), [0,1]; use existing LeNet inference.
    pass
```

## 完整复现

本地代码根目录为 `D:/code/FPGA/FPGA-main/FPGA-main`，原图实际位于外层 `../data/data/raw`。以下命令始终从代码根目录执行。为复现创建新结果目录，禁止覆盖已冻结结果。

```powershell
python -m unittest discover -s level2/tools -p "test*.py"
python level2/tools/preprocessing_experiment.py --input "../data/data/raw" --output level2/results/replication_v1
python level2/tools/evaluate_robust.py dev --output level2/results/replication_v1
python level2/tools/evaluate_robust.py final --output level2/results/replication_v1
python level2/tools/report_robust.py --output level2/results/replication_v1
```

`dev` 只评估开发集，按正确数选择不加粗/加粗两个最终候选，平局选更简单的不加粗方案。早期步骤只用于消融。`final` 核验图像、参数、配置和预处理源码哈希，不允许改完规则后偷偷复用冻结记录。测试集新方法评估只运行一次。重复运行同一冻结版本只是复现，不能称作新的独立实验。

## HLS 交接

本机未找到可用 HLS 命令或旧脚本指定安装目录，生成了输入和对照文件供队友运行。预处理在 CPU 上执行，不是 HLS 硬件实现。

```powershell
$env:VIVADO_HLS_ROOT="实际安装目录"
cmd /c level2\run_robust_hls.bat
```

该脚本固定使用 `preprocessing_v1` 结果目录，默认只跑 C Simulation，使用新的 HLS 工作目录，保留现有历史结果。它不伪造工具版本，不调用旧的 `finalize-hls`，不自动把任务状态改成完成。

`preprocessed_accepted.bin` 含 495 张通过预处理检测的图片。`hls_input_mapping.csv` 保留它们对应的 503 张原样本索引和划分。`python_results.csv` 是浮点输出，`integer_reference_results.csv` 是按 W16 AP_RND/AP_SAT 逐次累加模拟的整数参考，不是 HLS 输出。

HLS 仅接受集的识别率以 495 为分母，但正式端到端正确率应以 503 为分母，8 张拒识计错。`threshold=0` 仅关闭比较工具的门槛，不代表准确率达标。新的 HLS 对照完成前，不沿用旧版“一致率 100%”结论。

## 有意保留的限制

算法假设背景占多数、单一主体数字、明暗反差足够。严重阴影、复杂背景、多数字、相连符号、过近的大目标仍可能失败。最大的遗留问题是 9，完整集 23/50，留出集 5/15。还缺真实背景负样本和按拍摄者/批次分组的全新测试集。本次不继续利用已揭开的留出结果调参。
