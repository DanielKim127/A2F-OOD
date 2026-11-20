# 原子与化学键特征定义

按用户提供的属性表实现。每个属性单独 one-hot，再按表格顺序拼接。分类属性最后一项为 UNKNOWN；布尔属性严格使用 `[True, False]`，只有两维。

## 原子：157 维

区间使用 Python 切片的左闭右开表示。

| 属性 | 维度 | 拼接区间 | 类别顺序 |
|---|---:|---|---|
| Atom Type | 119 | `[0:119]` | 原子序数 1–118，UNKNOWN |
| Degree | 7 | `[119:126]` | 0, 1, 2, 3, 4, 5, UNKNOWN |
| Formal Charge | 6 | `[126:132]` | -1, -2, 1, 2, 0, UNKNOWN |
| Radical Electrons | 6 | `[132:138]` | 0, 1, 2, 3, 4, UNKNOWN |
| Chirality Tag | 5 | `[138:143]` | CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER, UNKNOWN |
| Implicit Hydrogens | 6 | `[143:149]` | 0, 1, 2, 3, 4, UNKNOWN |
| Hybridization | 6 | `[149:155]` | SP, SP2, SP3, SP3D, SP3D2, UNKNOWN |
| Aromatic | 2 | `[155:157]` | True, False |

```text
119 + 7 + 6 + 6 + 5 + 6 + 6 + 2 = 157
```

隐式氢使用 RDKit `GetNumImplicitHs()`，而非总氢数。例如 `C` 的隐式氢为 4，而 `[CH4]` 的 4 个氢是显式标注，隐式氢为 0。截图中的 `CHI_TETRAHEDRAL_C` 按 RDKit 正式枚举名 `CHI_TETRAHEDRAL_CW` 实现。

原来特征中的原子质量、原子是否在环中、归一化总氢数不在该表内，已从原子特征中去除。芳香性改为两位 one-hot，新增独立的自由基电子数量字段。对所有元素使用原子序数 1–118，不再只覆盖原来的常用元素列表。

## 化学键：14 维

| 属性 | 维度 | 拼接区间 | 类别顺序 |
|---|---:|---|---|
| Bond Type | 5 | `[0:5]` | SINGLE, DOUBLE, TRIPLE, AROMATIC, UNKNOWN |
| Bond Stereo | 5 | `[5:10]` | STEREONONE, STEREOANY, STEREOZ, STEREOE, UNKNOWN |
| Conjugated | 2 | `[10:12]` | True, False |
| In Ring | 2 | `[12:14]` | True, False |

```text
5 + 5 + 2 + 2 = 14
```

键总宽度与旧实现相同，但字段顺序、立体化学类别和布尔编码均发生改变，所以旧键编码不兼容。非边位置和 padding 保持全零，由邻接矩阵及掩码排除；全零不表示一个真实的 UNKNOWN 化学键。

## 隐藏维度与网络适配

通用默认配置及全部十个正式数据配置使用 `hidden_dim=256`、`attention_heads=4`。输入特征宽度通过线性投影转换到隐藏空间，隐藏维度不需要等于 157 或 14；256 能被 4 个注意力头整除。

| 模块 | 维度 |
|---|---|
| 原子输入投影 | `157 → 256` |
| 每层 GINE 的键投影 | `14 → 256` |
| GINE 更新 MLP | `256 → 512 → 256` |
| SMILES Token Embedding | `vocab_size → 256` |
| 每个注意力头 | `256 / 4 = 64` |
| Transformer 前馈层 | `256 → 1024 → 256` |
| 共有候选 MLP | `768 → 512 → 256` |
| 原子/全局门控 | `514 → 256 → 2` |
| 自适应融合 MLP | `1024 → 512 → 256` |
| 性质预测头 | `256 → 256 → num_outputs` |

小型 smoke 和单元测试允许显式指定较小的隐藏维度，用于接口与回归验证，输入特征仍然是 157/14 维。

## 版本与验证

特征版本为 `table_atom157_bond14_v1`，定义保存在 `src/a2f_ood/data.py`。新的 runtime.json 和 best/last checkpoint 都保存特征版本、宽度及各属性维度。加载不兼容特征版本或旧 42 维原子输入的 checkpoint 时会给出明确错误，避免把旧的 14 维键向量误当成当前编码使用。

原 42 维、128 隐藏维度模型需要使用其源码快照进行历史复现；当前正式模型应从头重训。CPU/CUDA 测试覆盖 157/14 维输入、256 维融合、前向反向传播、隐式与显式氢区别、类别顺序、UNKNOWN 回退和旧 checkpoint 拦截。
