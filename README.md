# A2F-OOD

## 数据

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

## 模型

1. **图编码**：GINE 按指定属性表使用 157 维原子特征和 14 维键特征，正式模型分别投影到 256 维隐藏空间，再按真实边进行聚合。原子/键类别、顺序、UNKNOWN 和维度详情见 [FEATURE_SCHEMA.md](FEATURE_SCHEMA.md)。预处理存储紧凑边列表，批处理再构建 padded 张量。
2. **SMILES 编码**：化学 token 扫描器 + Transformer。保留分支、环闭合、键和手性语法；映射编号从输入 token 中去除，只作为对应关系的旁路信息。
3. **原子对齐**：先规范化并重新解析分子，建立固定图原子顺序，再生成 mapped SMILES。随机序列改变遍历方式，atom map 仍指向固定图原子。
4. **局部融合**：原子对的和、乘积、绝对差经过 MLP 得到共有候选，随后计算两种私有残差，通过门控形成原子融合表示。
5. **全局融合**：图节点与全部 SMILES token 做双向跨注意力；图级和序列级表示经过全局门控后，与局部池化结果一起预测性质。
6. **跨模态辅助任务**：图向量预测 SMILES 向量、SMILES 向量预测图向量，目标一侧停止梯度；可选图条件 Transformer Decoder 重构同一分子的 canonical SMILES。
7. **性质预测**：支持二分类、多标签、多分类、回归，以及两个单模态辅助头。

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

