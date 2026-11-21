# A2F-OOD

## 介绍
OOD分布外分子性质预测模型


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
