# A2F-OOD：2D 分子图与 SMILES 双视图融合

本项目已经实现数据预处理、网络、辅助损失、GroupDRO、训练、模型保存、推理、评估、数据下载和消融实验配置生成。当前模型从头训练，不依赖预训练权重，也不需要 3D 数据或 torch-geometric。

## 环境与快速运行

当前工作区已建立可运行环境 `.venv`。它复用本机 `py39` 环境中的 PyTorch 1.12.0/CUDA，并在项目环境内安装 RDKit、scikit-learn 等依赖；没有修改原环境。已验证 CPU 和 NVIDIA RTX 3060 上的运行。项目不依赖名为 MSPF_MPP 的环境；如果以后找到该环境，也可在其中安装本项目。

在项目根目录运行：

~~~powershell
# 小型虚构数据：完整训练与输出，用来验证管线
.\.venv\Scripts\python.exe -m a2f_ood.train --config configs/smoke.yaml

# 正式数据预处理检查：不训练网络
.\.venv\Scripts\python.exe -m a2f_ood.train --config configs/datasets/drugood_ec50_assay.yaml --validate-only

# 正式训练，默认自动选择 CUDA
.\.venv\Scripts\python.exe -m a2f_ood.train --config configs/datasets/drugood_ec50_assay.yaml
~~~

在新的计算机上，建议 Python 3.10/3.11，先按设备安装 PyTorch，再安装本项目：

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
~~~

安装后也可以使用 `a2f-train`、`a2f-predict`、`a2f-evaluate` 三个命令。配置中相对路径以包含该配置的项目根目录为基准，不依赖终端当前目录。

## 官方数据

已下载并转换下列数据，原始 JSON 总计 199,534,179 字节（约 190 MiB）：

| 数据 | 版本 | 训练 | OOD 验证 | OOD 测试 |
|---|---|---:|---:|---:|
| DrugOOD IC50 | assay | 34,953 | 19,475 | 19,463 |
| DrugOOD IC50 | scaffold | 22,025 | 19,478 | 19,480 |
| DrugOOD IC50 | size | 37,497 | 17,987 | 16,761 |
| DrugOOD EC50 | assay | 4,978 | 2,761 | 2,725 |
| DrugOOD EC50 | scaffold | 2,743 | 2,723 | 2,762 |
| DrugOOD EC50 | size | 5,189 | 2,495 | 2,505 |
| ADMEOOD EC50 | assay/core | 12,129 | 298 | 295 |
| ADMEOOD EC50 | scaffold | 11,723 | 583 | 583 |
| ADMEOOD KI | assay/core | 23,487 | 434 | 431 |
| ADMEOOD KI | scaffold | 31,110 | 1,207 | 1,204 |

这些是**官方原始划分的记录数**。网络预处理还有原子数和 token 长度上限，被排除的记录会写入报告；正式比较必须披露实际保留数量。

- `data/raw/drugood/`：官方 LBAP core IC50/EC50 JSON，分 assay/scaffold/size。LBAP 适用于只输入配体的分子性质任务。
- `data/raw/admeood/`：作者发布的 `label_EC50/KI_core/scaffold` JSON。core 在本项目配置中称作 assay；不是另一个独立的新划分。使用的是论文中的 label 变体，不与 confi 变体混用。
- `data/processed/drugood/` 与 `data/processed/admeood/`：训练用 CSV、五划分完整 CSV、元数据和审计结果。
- `configs/datasets/`：十份可直接运行的配置。
- `configs/benchmark_downloads.json`：固定的官方地址、文件大小、SHA256、ZIP member/CRC 等来源信息。

主 CSV 严格将官方 `train/ood_val/ood_test` 映射为 `train/valid/test`，保留官方 `cls_label` 和 `domain_id`，不重新划分或计算活性阈值。原始五个集合完整保存在 `.all_splits.csv`，**不要直接把它当成训练输入**。特别是 ADMEOOD scaffold 的某些 IID 集与训练集存在重复，已保留并记录，不混入当前 OOD 主实验。

已完成十个版本的规范化分子审计：选用的三个 OOD 训练/验证/测试集合之间均无相同规范化分子的交叉重叠。来源：
[DrugOOD](https://github.com/tencent-ailab/DrugOOD)、[ADMEOOD](https://github.com/qweasdzxc-wsy/ADMEOOD)。

重新获取、校验或转换：

~~~powershell
# 已存在文件也会校验大小及 SHA256；仅通过网络补齐缺失文件
.\.venv\Scripts\python.exe scripts/fetch_benchmarks.py
.\.venv\Scripts\python.exe scripts/fetch_benchmarks.py --list
.\.venv\Scripts\python.exe scripts/fetch_benchmarks.py --dataset drugood-ic50-assay

# 原始 JSON 已存在时重新转换，默认不覆盖已有输出
.\.venv\Scripts\python.exe scripts/prepare_datasets.py --raw-dir data/raw --output-dir data/processed --overwrite
~~~

DrugOOD 下载器使用 HTTP Range 从官方 ZIP 取出指定成员，避免获取整个约 1.16 GB 压缩包；校验成员 CRC 和展开文件 SHA256。服务器不支持 Range 时会明确报错，不会自动下载整个包。通用单文件 HTTPS/Drive 下载工具另见 `scripts/download_datasets.py --help`。

## 模型实现

1. **图编码**：GINE 按指定属性表使用 157 维原子特征和 14 维键特征，正式模型分别投影到 256 维隐藏空间，再按真实边进行聚合。原子/键类别、顺序、UNKNOWN 和维度详情见 [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md)。预处理存储紧凑边列表，批处理再构建 padded 张量。
2. **SMILES 编码**：化学 token 扫描器 + Transformer。保留分支、环闭合、键和手性语法；映射编号从输入 token 中去除，只作为对应关系的旁路信息。
3. **原子对齐**：先规范化并重新解析分子，建立固定图原子顺序，再生成 mapped SMILES。随机序列改变遍历方式，atom map 仍指向固定图原子。
4. **局部融合**：原子对的和、乘积、绝对差经过 MLP 得到共有候选，随后计算两种私有残差，通过门控形成原子融合表示。
5. **全局融合**：图节点与全部 SMILES token 做双向跨注意力；图级和序列级表示经过全局门控后，与局部池化结果一起预测性质。
6. **跨模态辅助任务**：图向量预测 SMILES 向量、SMILES 向量预测图向量，目标一侧停止梯度；可选图条件 Transformer Decoder 重构同一分子的 canonical SMILES。
7. **性质预测**：支持二分类、多标签、多分类、回归，以及两个单模态辅助头。

跨模态预测误差输入融合门控时会 detach，避免门控通过误差分支反向操纵该信号。误差和随机 SMILES 预测方差是**分歧诊断**，尚不代表校准后的错误概率；代码不声称已识别真实因果结构。

模型不会生成不同结构的“反事实分子”并自动继承原标签。图到 SMILES 解码仅恢复同一分子视图。性质推理会显式排除 teacher-forcing 目标，解码器只在训练或显式生成评估时使用。贪心解码不保证化学有效，输出有效率和分子 exact match 由 RDKit 另行检验。

## 训练目标与配置

正式二分类/多标签配置使用 `loss.classification_weighting: balanced`，仅根据预处理后保留的训练标签统计正负比例，设置 `w_positive=N/(2*N_positive)`、`w_negative=N/(2*N_negative)`。融合主头和两个单模态辅助头使用同一套类别加权 BCE，GroupDRO 在该逐样本损失上作组重加权。运行目录的 `class_weighting.json` 保存计数和权重，checkpoint 保存拟合后的权重；日志同时记录普通与加权 BCE。详细定义、缺失标签及单类别处理见 [CLASS_BALANCED_BCE.md](CLASS_BALANCED_BCE.md)。

总目标是稳健监督风险加五个辅助项：

- `lambda_atom_alignment`：对应原子的双向 InfoNCE，对有效原子采样/截取上限避免平方内存过大。
- `lambda_serialization_consistency`：随机 SMILES 间的表示与预测一致性。
- `lambda_cross_modal`：双向潜在表示预测。
- `lambda_generation`：图条件 SMILES 生成交叉熵。
- `lambda_unimodal`：两个单模态性质头。
- `use_group_dro`：在训练环境上使用指数权重 GroupDRO，否则使用 ERM。

当前默认先用 `loss.supervised_warmup_epochs=2` 轮训练主预测任务，再用 `loss.auxiliary_warmup_epochs=10` 轮逐步加入辅助项。原子对齐、生成和单模态监督的目标权重分别为 0.02、0.01、0.05。

正式数据配置采用 `group_dro_update_mode=epoch`：按训练组频率初始化环境权重，使用 `q(group)/p_train(group)` 估计重加权风险，每轮按各环境的平均损失更新一次，避免大组因出现频率高而多次累积权重。重要性权重上限为 3，主监督目标保留 75% ERM、最多加入 25% 重加权风险；前 5 轮不启用对抗重加权，再用 5 轮升至目标强度。`batch` 模式保留旧实现接口，用于复现实验。

学习率默认为 1e-4，先 warmup 3 轮，再按验证指标平台期减半，最低 1e-6。选模默认使用 3 个固定 SMILES 视图，最终报告使用 5 个视图；`monitor_min_delta` 只控制早停和学习率调度的显著改善判断，最佳 checkpoint 仍按原始验证分数保存。词表和 GroupDRO 组编号仅由训练集拟合；缺失标签使用掩码，测试指标在模型选择完成后计算。

2026-11-19 的训练稳定性分析与修改记录见 [TRAINING_STABILITY_REPORT.md](TRAINING_STABILITY_REPORT.md)。

可通过重复 `--set` 覆盖参数，例如：

~~~powershell
.\.venv\Scripts\python.exe -m a2f_ood.train --config configs/datasets/drugood_ec50_assay.yaml --set training.batch_size=16 --set experiment.seed=43 --set experiment.output_dir=outputs/ec50_assay_seed43
~~~

任务设置：

| task_type | 标签格式 | 其他要求 |
|---|---|---|
| binary | 一个 0/1 列 | 空白/NaN 为缺失 |
| multilabel | 多个 0/1 列 | 每个任务分别掩码 |
| regression | 一个或多个数值列 | MSE 或 Huber |
| multiclass | 一个整数类别列 | 类别从 0 开始，设置 data.num_classes |

默认骨架示例 `data/example.csv` 使用虚构标签，只用于管线演示。`configs/smoke.yaml` 是小网络与 5 轮训练配置，不能把其得分用作方法效果证据。

## 输出与推理

每次训练写入：

- `best.pt`：模型参数、训练配置、训练词表、组映射、优化器及 GroupDRO 状态。
- `last.pt`：最近完成轮次的模型、优化器、学习率调度器和 GroupDRO 状态；不代表最佳验证模型。
- `history.json`、`train.log`：每轮原始损失、实际加权辅助损失、学习率、梯度范数、裁剪比例、GroupDRO 诊断与验证指标。
- `valid_predictions.csv`、`test_predictions.csv`：性质预测、单模态预测、跨模态误差、随机 SMILES 方差。
- `metrics.json`：任务指标、ECE/Brier 等分类校准指标、各环境表现、最差环境和分歧分桶表现。
- `split_manifest.json`、`overlap_audit.json`、`preprocessing_summary.json`、`skipped_rows.json`：实际样本、排除及重叠审计。
- `vocabulary.json`、`config.yaml`、`runtime.json`：复现实验的信息。

没有同时出现正负例的环境不计算 ROC-AUC/AP，JSON 中记为 null，汇总时只平均可定义的组。全体无标签的推理也正常输出，指标记为 null。

~~~powershell
# 输入 CSV 可以没有性质标签；存在 group 列时会保留
.\.venv\Scripts\python.exe -m a2f_ood.predict --checkpoint outputs/smoke/best.pt --csv data/example.csv --output outputs/smoke/new_predictions.csv --views 5

# 仅在 checkpoint 启用 decoder 时支持生成评估
.\.venv\Scripts\python.exe -m a2f_ood.predict --checkpoint outputs/smoke/best.pt --csv data/example.csv --output outputs/smoke/generated.csv --generate-smiles

# 从已导出 CSV 重算二分类指标
.\.venv\Scripts\python.exe -m a2f_ood.evaluate --predictions outputs/smoke/test_predictions.csv --task-type binary --label-columns label --prediction-columns prediction --group-column group --output outputs/smoke/recomputed.json
~~~

默认要求 train/valid/test 中相同 canonical molecule 不交叉。仅复现确实有重复的外部官方协议时，可以显式设置 `data.overlap_policy=allow_official`；该策略只支持预分配划分，会保存/警告重复情况，不删除记录。本项目下载的十份 OOD 主 CSV 无须开启。

## 消融实验

~~~powershell
# 生成实验配置，默认三种种子，不立即训练
.\.venv\Scripts\python.exe scripts/run_ablation.py --config configs/datasets/drugood_ec50_assay.yaml

# 选择部分变体并实际执行
.\.venv\Scripts\python.exe scripts/run_ablation.py --config configs/datasets/drugood_ec50_assay.yaml --variants full no_reliability_gate no_decoder --seeds 42 43 44 --run
~~~

提供 full、graph、smiles、concat、关闭原子对齐、关闭跨注意力、关闭门控、关闭跨模态损失、关闭序列一致性、关闭 GroupDRO、关闭 decoder 等变体。graph/smiles/concat 基线关闭所有辅助损失和 GroupDRO，比较普通监督训练；模块消融则保留其他设定。正式实验应使用相同预算、多种随机种子，并披露长度筛选，不根据测试表现选超参。

## 代码目录与测试

~~~text
src/a2f_ood/
  config.py       配置合并、校验、路径解析
  data.py         RDKit、tokenizer、词表、随机视图、批处理
  splits.py       预分配/scaffold/size/environment/random 拆分及审计
  model.py        GINE、Transformer、对齐、融合、生成
  losses.py       掩码任务损失、辅助目标、GroupDRO
  metrics.py      概率/回归指标与最差环境
  engine.py       数据准备、训练一步、推理、checkpoint
  train.py        训练命令
  predict.py      已训练模型推理命令
  evaluate.py     CSV 指标命令
  datasets.py     官方原始数据转换和通用下载
scripts/
  fetch_benchmarks.py    固定官方数据下载与完整性校验
  download_datasets.py   通用 HTTPS / Google Drive 下载
  prepare_datasets.py    官方 JSON 转换
  run_ablation.py        消融配置生成及运行
~~~

~~~powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
.\.venv\Scripts\python.exe -m pytest -q
~~~

测试覆盖手性/同位素与原子映射、padding 和图原子置换不变性、缺失标签、分组风险、decoder 因果掩码、四种任务的完整训练和模型重载、无标签推理、下载完整性和官方格式转换。完整科学基准与方法优越性仍需后续正式实验验证。

