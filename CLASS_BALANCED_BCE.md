# 按训练集正负比例加权的 BCE

正式二分类和多标签分类配置启用 `loss.classification_weighting: balanced`。
每次训练在化学预处理和数据划分后，仅根据保留的训练集有标签样本，逐任务统计
负例数 `N_negative`、正例数 `N_positive`，并令 `N=N_negative+N_positive`：

```text
w_negative = N / (2 * N_negative)
w_positive = N / (2 * N_positive)
L = -w_positive*y*log(sigmoid(z)) - w_negative*(1-y)*log(1-sigmoid(z))
```

实现使用数值稳定的 `binary_cross_entropy_with_logits`，再按真实类别乘以权重。
两类在训练集中的总权重均为 `N/2`，全部有标签样本的平均权重为 1。
正负相对权重等于 `pos_weight=N_negative/N_positive`；再乘以 `w_negative`
即得到同一损失。正例多于负例时，负例会得到更高权重。

权重不按批次重算，不读取验证集或测试集标签。多标签任务按列独立统计并忽略缺失值；
若某列训练标签只有一个类别或完全没有标签，使用 `[1,1]` 中性权重并记录警告。
融合主头、图辅助头、SMILES 辅助头均使用相同权重；GroupDRO 在类别加权后的
逐分子损失上作原有的组重加权。表征、生成和一致性损失保持其原有定义。

每次运行保存 `class_weighting.json`，包含训练计数、权重与回退记录；拟合后的
`loss.binary_class_weights` 同时保存在运行配置和 checkpoint 中。即便以保存的配置
开始新训练，也会在新训练集重新拟合权重。旧配置缺省使用 `none`，可显式设置
`loss.classification_weighting=none` 重现普通 BCE。

训练日志中 `train.task` 为实际优化的类别加权 BCE，`train.task_unweighted`
额外记录普通 BCE；`robust` 在此基础上混合 GroupDRO。评估的 `task_loss` 和
`log_loss` 继续报告普通损失，新增 `class_weighted_task_loss` 使用训练拟合的固定权重。
训练计算每个视图的损失再平均；评估先平均多视图概率，因此二者并非相同口径。
ROC-AUC、AP、Accuracy、ECE 仍以直接 sigmoid 预测计算，无额外概率或阈值校正。

加权分类目标改变了误分类代价，sigmoid 输出未必是原始分布下的校准概率。
实验同时报告 ROC-AUC、AP、普通 Log loss、ECE 及阈值指标，避免仅依据 Accuracy 判断。

验证：127 项测试通过，包括 CPU/CUDA 上权重归一、类别梯度总量平衡、缺失标签、
单类别回退、三类分类头一致性、GroupDRO 组合，以及权重保存和评估的完整流程。
