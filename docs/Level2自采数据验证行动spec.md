# Level 2 自采数据验证行动 spec

目标：将根目录 `data/` 接入已有 Level 2 流程，形成可复现的 Python 与 HLS 定点评估及中期展示材料。按下面顺序完成；本次沿用 Level 1 的模型、权重和 W16 格式，不开展训练、位宽扫描或架构优化。

## 依据与已知情况

- 先读 `docs/当前验收状态.md`、`level2/README.md`；课程要求详见 `docs/任务要求与LeNet路线核查.md` 的 Level 2 条目。原始任务书未入库，要求依据来自仓库核查记录。
- Level 1 已有完整 MNIST 对照和综合证据；课程仅要求仿真与综合，无需上板。Level 2 需要真实采集图像、完整预处理、准确率变化分析及无目标背景测试；没有已确认的 Level 2 90% 门槛。
- 当前输入位于 `data/data/raw/0` 到 `9`；忽略 `__MACOSX`、`._*`、`.DS_Store` 等附属文件。根目录 `/data/` 已被 Git 忽略。
- `level2/tools/level2_validation.py` 的 `evaluate-folder` 只读平铺目录且要求文件名以 `标签_` 开头，不能直接处理当前结构。复用其预处理、`raw_resize` 以及 Level 1 的推理和 blob 读写函数。
- 文件名包含 `zimage_topdown_v1_digit...`，来源待核实。不能仅凭目录名或文件名认定它是组员实拍，也不能据此认定它是生成图。

## 1. 核对数据

生成稳定排序的清单，至少记录相对路径、标签、尺寸、文件 SHA256、读取状态。标签取直接父目录 `0..9`；检查文件名中存在的 digit 标签是否冲突，冲突明确报错。统计每类有效数、损坏文件、重复内容与背景样本数，所有排除项列明原因；保留原文件。

检查已有来源说明；若不足，向用户询问实拍/生成来源、采集方式和背景图位置，同时继续数据适配与技术评估。来源未确认时报告写“来源待确认的数据集”，正式真实自采验收保持待完成。若是生成图，作为补充压力测试单独报告。

完成条件：有效图像均能解码，标签合法，清单与实际评估数量对应；背景图片单独分组，不赋予数字标签。

## 2. 最小适配并跑 Python

在现有 Level 2 工具中增加一个支持分类子目录的入口，保留旧入口可用。全部新结果放到 `level2/results/self_collected/`；重复实验使用独立子目录，避免覆盖历史证据。新增 blob、缓存和完整尺寸图片加入精准的忽略规则，保留可评审的摘要、CSV 和小图。

固定参数源 `level1/data/lenet_accuracy_1.bin`，在同一清单上运行直接缩放和完整预处理两组浮点推理。先每类一张冒烟检查，再跑全量有效数字图片。预处理结果必须是有限值的 float32、28×28、范围 [0,1]，极性与 MNIST 相同。

输出以下内容：

- `manifest.csv`、`predictions.csv`：保留 index 到原图的映射、真实标签及两种预处理的预测。
- `summary.json`：样本数、每类数量、两组正确数/准确率、预处理收益（百分点），参数文件哈希、Git commit 和本地改动情况。
- `python_results.csv` 与 `preprocessed_self_collected.bin`：使用现有 `write_result_csv`、`write_lenet_blob` 格式，样本顺序完全相同。
- 两组混淆矩阵、一张原图/直接缩放/预处理对比图、典型失败案例。

新增少量测试，覆盖分类目录标签、元数据过滤、空目录/非法标签失败、blob 回读后的顺序与像素一致性。运行：

```sh
python3 -m unittest discover -s level2/tools -p 'test_*.py'
python3 -m unittest discover -s level1/tools -p 'test_lenet_validation.py'
```

完成条件：测试通过，所有数字样本均有结果，指标可从 CSV 重算。低准确率也是有效结果；先分析裁剪、背景、笔迹等失败原因。如调整预处理，保留调整前结果并披露调参使用的数据。

## 3. 同批数据跑 HLS C 仿真

需要 Vivado HLS 或兼容的 Vitis HLS C Simulation；新增图片本身不要求重新综合、启动 Vivado GUI、布局布线或上板。先重新检查 `vivado_hls`、`vitis_hls`，以及 Vitis 安装目录中由 `loader -exec vitis_hls` 启动的内部 HLS 程序；可用时执行，不可用时交付可运行脚本并将 HLS 标为待执行。

复用 `level1/run_hls.tcl` 和其 testbench。创建单独的运行脚本，在启动前检查 blob 存在且样本数与清单一致，避免 Tcl 缺文件时回退到 smoke test。仓库根目录的 POSIX 命令模板如下（先加载实际安装的 HLS 环境）：

```sh
LENET_DATA_W=16 LENET_SKIP_SYNTH=1 LENET_SKIP_CSIM=0 \
LENET_ACCURACY_THRESHOLD=0 \
LENET_HLS_WORKSPACE="$PWD/level2/hls_work/self_collected_w16" \
LENET_ACCURACY_BLOB="$PWD/level2/results/self_collected/preprocessed_self_collected.bin" \
LENET_RESULT_CSV="$PWD/level2/results/self_collected/hls_results.csv" \
vivado_hls -f level1/run_hls.tcl

python3 level1/tools/lenet_validation.py compare \
  --float-results level2/results/self_collected/python_results.csv \
  --hls-results level2/results/self_collected/hls_results.csv \
  --report level2/results/self_collected/python_hls_comparison.json \
  --mismatches level2/results/self_collected/python_hls_mismatches.csv \
  --threshold 0
```

Windows 环境提供等价 `.bat`，工具路径按实际安装配置。保留完整命令、工具版本、退出码和 CSim 日志；按需精确放行该日志的 Git 忽略规则。

`threshold=0` 仅用于取消 MNIST 90% 门槛对自采数据的阻断。比较器的 PASS 仅表示准确率门槛通过：还必须核对输出行数、index、标签，并独立报告浮点/HLS 准确率、预测一致率与不一致数。浮点与定点 logits 不要求逐位相等；出现预测差异时检查前两类分数间隔，必要时用 `layer_validation` 的同格式定点参考定位，未解释差异标为待定位。

不要直接调用 `run_stress_hls_windows.bat` 或 `finalize-report` 作为自采报告流程：前者绑定旧压力数据，后者写死了 55.10% 并错误假设两种准确率相等。新报告分别引用实际指标。

完成条件：全量数字输入实际完成 CSim、结果可追溯，差异已说明或明确列为未解决。Python 成功不能替代 HLS 成功。此次证据属于 Level 1 W16 内核，不自动覆盖 row_cache 或混合精度变体。

## 4. 背景测试与中期交付

背景图独立报告数量、预处理非零输出情况及分类器输出。当前十分类模型总会输出数字，零图也不代表具有背景拒识能力。如报告“误检率”，必须先说明实际采用的拒识规则和分母，同时统计数字样本误拒率；没有拒识规则就明确说明能力缺口。缺少真实背景图时列为待补，程序生成的空白图仅用于单元测试。

写 `experiment_report.md`：数据来源与数量、预处理流程、已有 MNIST 基线（注明历史数据及配置）、本次直接缩放/预处理/HLS 准确率、混淆矩阵、失败案例、背景测试、复现命令及剩余事项。引用历史综合和 RTL 结果时注明对应实现与样本范围。

更新 `level2/README.md` 的真实可执行命令，并按证据更新 `docs/当前验收状态.md`。最终交付代码、测试结果、报告及一张适合中期展示的结果概览图；总结 Python、HLS、数据来源、背景测试各自完成状态。

范围边界：本任务默认只修改 Level 2 数据接入、测试和文档。若发现必须修改可综合 C++、定点格式或硬件接口才能解决的问题，先报告原因与拟改动范围；此类变更需另行安排 CSim、综合及必要的 RTL 回归。仅新增测试输入无需重跑已有完整 MNIST、九档位宽扫描或 RTL 全量仿真。
