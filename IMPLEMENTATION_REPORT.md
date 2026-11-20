# 实现与验证记录

日期：2026-11-18。

2026-11-19 更新：针对真实训练中融合预测回落的问题，修正 GroupDRO 的组出现频率偏差和风险估计，加入 ERM 锚定、重要性上限、辅助任务分阶段启用与学习率调度，修复迁移后的项目导入路径，并增加训练诊断和最近轮次 checkpoint。按指定属性表将原子特征改为 157 维、键特征更新为新的 14 维定义，正式隐藏维度改为 256，增加特征版本与 checkpoint 检查。当前全套测试为 118 passed；真实数据仅进行了预处理校验，完整实验保持暂停。详细证据及限制见 [TRAINING_STABILITY_REPORT.md](TRAINING_STABILITY_REPORT.md) 和 [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md)。下文的“81 passed”和两轮 smoke 等为 2026-11-18 的历史验证记录。

项目已实现从 CSV/RDKit 预处理到训练、checkpoint、预测、指标计算的完整流程，src 和 scripts 中没有 NotImplemented/TODO 占位。

## 已验证

- 全套自动测试：81 passed。
- 四种任务：二分类、多标签、回归、多分类均有两轮小型集成测试，覆盖缺失标签、GroupDRO、decoder、模型重载和输出。
- CPU 小样例训练完成；RTX 3060 GPU 两轮小样例训练完成。
- 真正官方数据的小批次（每份原始选取 train128/valid32/test32）：
  - DrugOOD EC50 assay：两轮 CUDA 训练、best.pt、预测及指标输出，保留 train128/valid31/test32。
  - ADMEOOD EC50 assay：两轮 CUDA 训练、best.pt、预测及指标输出，全部保留。
  - 结果位于 outputs/real_data_smoke/，仅证明真实格式与训练接口相容，不是完整基准实验。
- 另做虚构标签九个训练分子的过拟合检查：训练任务损失从 0.68627 降至约 0.0000202；验证表现未作为方法有效性的证据。
- 所有十份官方原始文件按固定清单验证大小/SHA256，DrugOOD ZIP 成员还校验 CRC。
- 本地依赖检查：No broken requirements found。
- 消融配置生成脚本已实际执行，输出于 outputs/ablation_preview/。

## 完整预处理检查

| 数据 | 官方 train/valid/test | 默认上限保留 train/valid/test | 排除 |
|---|---|---|---:|
| DrugOOD EC50 assay | 4978 / 2761 / 2725 | 4898 / 2638 / 2719 | 209 |
| ADMEOOD EC50 assay | 12129 / 298 / 295 | 12129 / 298 / 295 | 0 |

配置上限为256原子、512个SMILES token（含特殊token）。大分子或过长序列被跳过而不截断；每条被排除的记录保存在对应 outputs/<dataset>/skipped_rows.json，统计见 preprocessing_summary.json。正式对比时应统一数据过滤策略，或提高上限，并披露各拆分实际样本量。

## 数据范围

- DrugOOD：LBAP core IC50/EC50 × assay/scaffold/size，共六份。
- ADMEOOD：label EC50/KI × assay(core)/scaffold，共四份。
- 下载原始JSON共199,534,179字节，位于 data/raw/。
- 主CSV位于 data/processed/，只选官方 train/ood_val/ood_test；原始五拆分另存 all_splits.csv。
- 所选三个拆分间均无规范化分子重叠；ADMEOOD scaffold 的某些原始 IID 集与训练有重叠，元数据已记录，不混入主 OOD 实验。

## 解释范围

融合门控使用跨模态预测分歧，尚不是校准后的错误概率。模型是可运行的研究实现，没有完成全部十份数据的正式100轮多种子基准，也没有证据保证优于已有方法。正式实验配置、消融脚本和日志输出已备齐。图条件SMILES解码没有化学有效性保证，评估会单独计有效率和分子exact match。

当前本地项目环境 .venv 使用 Python3.9.16、PyTorch1.12.0/CUDA、NumPy1.23.5、RDKit2025.9.2，复用现有py39的PyTorch，并在项目环境中安装其他依赖。

