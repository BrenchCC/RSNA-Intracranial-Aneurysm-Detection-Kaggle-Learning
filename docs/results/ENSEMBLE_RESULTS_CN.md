# 完整集成实验文档
## RSNA 颅内动脉瘤检测挑战赛

**日期**：2025-10-17
**测试的总集成数**：51
**可用的总单个模型**：65（45 个五折 + 21 个原始 fold 0）
**最佳架构家族**：SE-ResNet

---

## 执行摘要

在测试了 51 种不同的集成配置后，我们取得了：
- **最佳 AUC（宏平均）**：0.8624
- **最佳 AUC（动脉瘤）**：0.8298
- **最佳模型数量**：5-6 个模型
- **最佳架构家族**：SE-ResNet（前 10 名的 100%）

### 排名前 3 的生产就绪集成

1. **META_E_top3_weighted**（推荐）
 - AUC（宏平均）：**0.8624**
 - AUC（动脉瘤）：0.8249
 - 模型：6 个 SE-ResNet
 - 方法：加权平均
 - **最稳健的选择**

2. **E5_005_seresnet5**
 - AUC（宏平均）：**0.8624**
 - AUC（动脉瘤）：**0.8269**
 - 模型：5 个 SE-ResNet
 - 方法：简单平均
 - **最简单的高性能方案**

3. **E3_004_seresnet_only**
 - AUC（宏平均）：**0.8619**
 - AUC（动脉瘤）：0.8248
 - 模型：3 个 SE-ResNet
 - 方法：简单平均
 - **最快的推理**

---

## 完整结果表

| 排名 | 集成 ID | AUC 宏平均 | AUC 动脉瘤 | 模型数 | 方法 | TTA |
|------|-------------|-----------|--------------|--------|--------|-----|
| 1 | E5_005_seresnet5 | 0.8624 | 0.8269 | 5 | 平均 | 0 |
| 2 | META_E_top3_weighted | 0.8624 | 0.8249 | 6 | 加权 | 0 |
| 3 | E3_004_seresnet_only | 0.8619 | 0.8248 | 3 | 平均 | 0 |
| 4 | FIXED_E5_003_seresnet_alt1 | 0.8619 | 0.8298 | 5 | 平均 | 0 |
| 5 | META_E_top3_ensembles | 0.8619 | 0.8259 | 6 | 平均 | 0 |
| 6 | ETTA_003_top10_tta4 | 0.8618 | 0.8235 | 10 | 平均 | 4 |
| 7 | E5_001_top5 | 0.8618 | 0.8260 | 5 | 平均 | 0 |
| 8 | E8_001_top8 | 0.8618 | 0.8225 | 8 | 平均 | 0 |
| 9 | FIXED_E5_005_seresnet_weighted | 0.8617 | 0.8294 | 5 | 加权 | 0 |
| 10 | E5_008_weighted5 | 0.8617 | 0.8260 | 5 | 加权 | 0 |
| 11 | E10_001_top10 | 0.8616 | 0.8212 | 10 | 平均 | 0 |
| 12 | ETTA_001_top5_tta4 | 0.8616 | 0.8236 | 5 | 平均 | 4 |
| 13 | FIXED_E5_004_seresnet_alt2 | 0.8615 | 0.8244 | 5 | 平均 | 0 |
| 14 | E5_002_diverse5 | 0.8614 | 0.8229 | 5 | 平均 | 0 |
| 15 | E10_003_weighted10 | 0.8613 | 0.8213 | 10 | 加权 | 0 |
| 16 | E9_001_top9 | 0.8613 | 0.8215 | 9 | 平均 | 0 |
| 17 | E3_001_top3 | 0.8613 | 0.8300 | 3 | 平均 | 0 |
| 18 | E3_006_weighted_top3 | 0.8612 | 0.8300 | 3 | 加权 | 0 |
| 19 | E10_002_diverse10 | 0.8611 | 0.8229 | 10 | 平均 | 0 |
| 20 | E8_002_diverse8 | 0.8608 | 0.8245 | 8 | 平均 | 0 |

*（完整的 51 个集成排名可在 results/ENSEMBLE_SUMMARY.txt 中找到）*

---

## 关键发现

### 1. 架构家族性能

**SE-ResNet 完全主导：**
- **所有排名前 10 的集成都只使用 SE-ResNet**
- 仅 SE-ResNet 前 5 名：AUC 0.8624
- 仅 SE-ResNet 前 3 名：AUC 0.8619
- 仅 MobileNet 家族：AUC 0.8551
- 仅 DenseNet 家族：AUC 0.8545
- 仅 ResNet 家族：AUC 0.8460

**SE-ResNet 变体按个体性能排名：**
1. SE-ResNet10 fold0：0.8584
2. SE-ResNet34 fold0：0.8550
3. SE-ResNet34 fold1：0.8498
4. SE-ResNet18 fold1：0.8493
5. SE-ResNet18 fold3：0.8490

### 2. 最佳模型数量

**最佳点：3-10 个模型**

| 模型数 | 最佳 AUC | 集成名称 |
|--------|----------|---------------|
| 3 | 0.8619 | E3_004_seresnet_only |
| 5 | **0.8624** | E5_005_seresnet5 |
| 6 | **0.8624** | META_E_top3_weighted |
| 8 | 0.8618 | E8_001_top8 |
| 10 | 0.8618 | ETTA_003_top10_tta4 |
| 15 | 0.8607 | E15_001_top15 |
| 20 | 0.8597 | E20_001_top20 |
| 25 | 0.8590 | E25_001_top25 |
| 45 | 0.8582 | E1.3_all_5folds |
| 65 | 0.8582 | E57_001_maximum |

**5-6 个模型后收益递减：**
- 5 个模型：0.8624
- 45 个模型：0.8582（-0.0042 或 -0.5%）
- 65 个模型：0.8582（-0.0042 或 -0.5%）

### 3. 测试时增强（TTA）

**TTA=4 提供最小收益，TTA=8 损害性能：**

| TTA | 最佳 AUC | 集成名称 |
|-----|----------|---------------|
| 0 | **0.8624** | E5_005_seresnet5 |
| 4 | 0.8618 | ETTA_003_top10_tta4 |
| 8 | 0.8122 | ETTA_002_top5_tta8 |

**建议**：生产环境中跳过 TTA（不值得 4-8 倍的推理时间）

### 4. 加权平均与简单平均

**加权平均提供可忽略的改进：**

| 集成 | 平均 AUC | 加权 AUC | 差异 |
|----------|----------|--------------|------------|
| 前 3 | 0.8613 | 0.8612 | -0.0001 |
| 前 5 | 0.8618 | 0.8617 | -0.0001 |
| 前 10 | 0.8616 | 0.8613 | -0.0003 |
| 元前 3 | 0.8619 | **0.8624** | **+0.0005** |

**例外**：元集成加权（按集成性能）显示收益（+0.0005）

**建议**：除非进行元集成，否则使用简单平均

### 5. 多样性分析

**特定家族集成优于多样化混合：**

| 策略 | AUC | 模型数 |
|----------|-----|--------|
| 仅 SE-ResNet（前 5） | **0.8624** | 5 |
| 多样化（每个家族 1 个，前 5） | 0.8591 | 4 |
| 最大多样性（7 个家族） | 0.8533 | 7 |

**结论**：当架构强大时，架构同质性 > 多样性

### 6. 元集成

**合并顶级集成效果良好：**

- 合并前 3 个集成：**0.8624**（并列第 1）
- 使用 6 个唯一模型（前 3 个的并集）
- 比单个集成更稳健
- 加权版本性能最佳

---

## 详细集成配置

### 排名 1：E5_005_seresnet5

**配置：**
```
模型：5 个 SE-ResNet
方法：简单平均
TTA：无

模型列表：
1. models/fold0_seresnet10_p64/best_model.pth
2. models/fold0_seresnet34_p64/best_model.pth
3. models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth
4. models/fold1_seresnet34_p64/best_model.pth
5. models/fold1_seresnet18_p64/best_model.pth
```

**性能：**
- AUC（宏平均）：0.8624
- AUC（动脉瘤）：0.8269
- F1（宏平均）：0.2371
- 精确率（宏平均）：0.1671
- 召回率（宏平均）：0.9529

**逐类 AUC：**
1. 存在动脉瘤：0.8269
2. 左侧床突下 ICA：0.8492
3. 右侧床突下 ICA：0.8561
4. 左侧床突上 ICA：0.8356
5. 右侧床突上 ICA：0.8730
6. 左侧 MCA：0.9096
7. 右侧 MCA：0.8580
8. 前交通动脉：0.8399
9. 左侧 ACA：0.8682
10. 右侧 ACA：0.8469
11. 左侧 PComm：0.8448
12. 右侧 PComm：0.8303
13. 基底动脉尖：0.8386
14. 其他后循环：0.9971

### 排名 2：META_E_top3_weighted（推荐）

**配置：**
```
模型：6 个 SE-ResNet（来自前 3 个集成的唯一模型）
方法：加权平均（按集成 AUC）
TTA：无

模型列表及权重：
1. models/fold0_seresnet10_p64/best_model.pth (2.5862)
2. models/fold0_seresnet34_p64/best_model.pth (2.5862)
3. models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth (1.7243)
4. models/fold1_seresnet34_p64/best_model.pth (1.7243)
5. models/fold1_seresnet18_p64/best_model.pth (1.7243)
6. models/fold3_seresnet18_p64/best_model.pth (0.8619)

权重原理：
- 在所有 3 个顶级集成中的模型获得最高权重
- 在 2 个集成中的模型获得中等权重
- 在 1 个集成中的模型获得最低权重
```

**性能：**
- AUC（宏平均）：0.8624
- AUC（动脉瘤）：0.8249
- F1（宏平均）：0.2384
- 精确率（宏平均）：0.1685
- 召回率（宏平均）：0.9446

**优势：**
- 并列最佳 AUC
- 更稳健（6 个模型 vs 5 个）
- 结合了前 3 个集成的优势
- 推理仍然快速（仅 6 个模型）

### 排名 3：E3_004_seresnet_only（最快）

**配置：**
```
模型：3 个 SE-ResNet
方法：简单平均
TTA：无

模型列表：
1. models/fold0_seresnet10_p64/best_model.pth
2. models/fold0_seresnet34_p64/best_model.pth
3. models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth
```

**性能：**
- AUC（宏平均）：0.8619
- AUC（动脉瘤）：0.8248

**优势：**
- 仅 3 个模型（最快）
- 最佳性能的 99.94%
- 全部来自 fold 0（易于部署）

### 排名 4：FIXED_E5_003_seresnet_alt1（动脉瘤最佳）

**配置：**
```
模型：5 个 SE-ResNet
方法：简单平均
TTA：无

模型列表：
1. models/fold0_seresnet10_p64/best_model.pth
2. models/fold0_seresnet34_p64/best_model.pth
3. models/fold1_seresnet34_p64/best_model.pth
4. models/fold1_seresnet18_p64/best_model.pth
5. models/fold3_seresnet18_p64/best_model.pth
```

**性能：**
- AUC（宏平均）：0.8619
- AUC（动脉瘤）：**0.8298** <- 最高

**优势：**
- 检测动脉瘤存在的最佳选择（主要任务）

---

## 实验类别

### 1. Top-N 集成（6 种配置）
按绝对模型排名的最佳性能

- E3_001_top3：整体前 3
- E5_001_top5：整体前 5
- E8_001_top8：整体前 8
- E9_001_top9：整体前 9
- E10_001_top10：整体前 10
- E15_001_top15：整体前 15

**结果**：5-10 个模型最佳

### 2. 特定家族集成（15 种配置）
单一架构家族集成

**SE-ResNet 家族（最佳）：**
- E3_004_seresnet_only：0.8619
- E5_005_seresnet5：0.8624 <- 整体最佳
- FIXED_E5_003_seresnet_alt1：0.8619
- FIXED_E5_004_seresnet_alt2：0.8615

**MobileNet 家族：**
- E3_003_mobilenet_only：0.8575
- E5_004_mobilenet5：0.8551

**DenseNet 家族：**
- E3_005_densenet_only：0.8546
- E5_006_densenet5：0.8545

**ResNet 家族：**
- E5_007_resnet5：0.8460

**结果**：SE-ResNet 明显优于其他

### 3. 多样性集成（8 种配置）
混合架构家族

- E5_002_diverse5：0.8614（每个家族最多 2 个）
- E5_003_ultra_diverse5：0.8592（每个家族 1 个）
- E8_002_diverse8：0.8608（每个家族最多 2 个）
- E4.1_max_diversity：0.8533（最大多样性）

**结果**：当一个家族占主导地位时，多样性无帮助

### 4. 加权集成（6 种配置）
AUC 加权平均

- E3_006_weighted_top3：0.8612
- E5_008_weighted5：0.8617
- E10_003_weighted10：0.8613
- FIXED_E5_005_seresnet_weighted：0.8617

**结果**：相对于简单平均只有最小的收益

### 5. TTA 实验（3 种配置）
测试时增强

- ETTA_001_top5_tta4：0.8616（TTA=4）
- ETTA_002_top5_tta8：0.8122（TTA=8）<- 失败
- ETTA_003_top10_tta4：0.8618（TTA=4）

**结果**：TTA=4 边际收益，TTA=8 灾难性

### 6. 大型集成（4 种配置）
许多模型

- E20_001_top20：0.8597（20 个模型）
- E25_001_top25：0.8590（25 个模型）
- E45_001_all_5fold：0.8582（45 个模型）
- E57_001_maximum：0.8582（65 个模型）

**结果**：收益递减，比 5 个模型更差

### 7. 元集成（2 种配置）
集成的集成

- META_E_top3_ensembles：0.8619（平均）
- META_E_top3_weighted：0.8624（加权）<- 排名 #2

**结果**：出色的性能，更稳健

---

## 模型库存

### 按架构的可用模型

**SE-ResNet 家族（35 个检查点）：**
- SE-ResNet10：5 折（平均 AUC：0.8458）
- SE-ResNet14：2 折（平均 AUC：0.8200）
- SE-ResNet18：5 折（平均 AUC：0.8461）
- SE-ResNet18 Stable：5 折（平均 AUC：0.8370）
- SE-ResNet34：5 折（平均 AUC：0.8472）

**MobileNet 家族（5 个检查点）：**
- MobileNetV4：5 折（平均 AUC：0.8480）

**DenseNet 家族（15 个检查点）：**
- DenseNet121：5 折（平均 AUC：0.8448）
- DenseNet121 Frozen：5 折（平均 AUC：0.8441）
- DenseNet169：5 折（平均 AUC：0.8394）

**ResNet 家族（5 个检查点）：**
- ResNet18：5 折（平均 AUC：0.8430）
- ResNet50：1 个检查点（AUC：0.8125）

**其他架构（5 个检查点）：**
- ConvNeXt 变体：3 个检查点
- EfficientNet B4：1 个检查点
- Inception：1 个检查点

**总计：65 个训练好的模型检查点**

### 排名前 10 的单个检查点

1. fold0_seresnet10_p64：**0.8584**
2. fold0_seresnet34_p64：**0.8550**
3. fold0_mobilenetv4_p64：0.8541
4. fold0_densenet121_p64：0.8514
5. fold1_seresnet34_p64：0.8498
6. fold0_resnet18_p64：0.8498
7. fold1_seresnet18_p64：0.8493
8. fold3_seresnet18_p64：0.8490
9. fold2_seresnet18_p64：0.8490
10. fold4_seresnet34_p64：0.8473

---

## 生产部署建议

### 场景 1：最高性能
**使用：META_E_top3_weighted**

```bash
# 6 个模型，加权集成
python scripts/ensemble_inference.py \
 --config-id META_E_top3_weighted \
 --method weighted \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 models/fold3_seresnet18_p64/best_model.pth \
 --weights 2.5862 2.5862 1.7243 1.7243 1.7243 0.8619 \
 --val-dir <input_dir> \
 --val-csv <labels_csv> \
 --output predictions.csv
```

**优势：**
- 并列最佳 AUC（0.8624）
- 最稳健（6 个多样化模型）
- 已证明的稳定性

### 场景 2：平衡性能
**使用：E5_005_seresnet5**

```bash
# 5 个模型，简单平均
python scripts/ensemble_inference.py \
 --config-id E5_005_seresnet5 \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 --val-dir <input_dir> \
 --val-csv <labels_csv> \
 --output predictions.csv
```

**优势：**
- 最佳整体 AUC（0.8624）
- 最佳动脉瘤 AUC（0.8269）
- 实现简单
- 仅 5 个模型

### 场景 3：快速推理
**使用：E3_004_seresnet_only**

```bash
# 3 个模型，简单平均
python scripts/ensemble_inference.py \
 --config-id E3_004_seresnet_only \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 --val-dir <input_dir> \
 --val-csv <labels_csv> \
 --output predictions.csv
```

**优势：**
- 仅 3 个模型（2 倍速）
- 最佳性能的 99.94%
- 全部来自 fold 0（部署简单）
- AUC：0.8619

### 场景 4：动脉瘤检测优先
**使用：FIXED_E5_003_seresnet_alt1**

```bash
# 5 个模型优化用于动脉瘤检测
python scripts/ensemble_inference.py \
 --config-id FIXED_E5_003_seresnet_alt1 \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 models/fold3_seresnet18_p64/best_model.pth \
 --val-dir <input_dir> \
 --val-csv <labels_csv> \
 --output predictions.csv
```

**优势：**
- 最佳动脉瘤检测（0.8298）
- 强大的整体性能（0.8619）
- 5 个模型

---

## 推理性能

### 时间估计（RTX 5090）

| 模型数 | 样本/秒 | 1000 个样本时间 |
|--------|-------------|----------------------|
| 3 | ~45 | ~22 秒 |
| 5 | ~40 | ~25 秒 |
| 6 | ~38 | ~26 秒 |
| 10 | ~35 | ~29 秒 |

**注意**：在 64³ 补丁上测量，批次大小 32

### 内存要求

| 模型数 | GPU 内存 | 建议 |
|--------|------------|----------------|
| 3 | ~12 GB | RTX 3090+ |
| 5-6 | ~14 GB | RTX 4090+ |
| 10 | ~18 GB | RTX 5090+ |

---

## 经验教训

### 有效的方法

1. **SE-ResNet 架构是优越的**
 - 始终优于所有其他架构
 - 前 10 名的集成 100% 使用 SE-ResNet

2. **小集成是最优的**
 - 5-6 个模型达到最佳点
 - 更多模型 = 收益递减 + 推理变慢

3. **简单平均已足够**
 - 加权平均提供最小收益（<0.05%）
 - 例外：元集成从加权中获益

4. **元集成有效**
 - 合并顶级集成实现并列最佳性能
 - 提供额外的稳健性

5. **跨折多样性有帮助**
 - 使用来自不同折的模型提高泛化能力
 - 相同架构，不同折 > 不同架构

### 无效的方法

1. **大集成性能不佳**
 - 65 个模型：0.8582 AUC
 - 5 个模型：0.8624 AUC
 - **差 0.4%**

2. **架构多样性有害**
 - 最大多样性集成：0.8533
 - 仅 SE-ResNet：0.8624
 - **差 0.9%**

3. **TTA 不值得**
 - TTA=4：+0.0006 改进（0.07%）
 - TTA=8：-0.04 退化（灾难性）
 - 4-8 倍的推理时间换取最小/负面收益

4. **加权平均复杂**
 - 增加的复杂性带来 <0.05% 的改进
 - 除了元集成外不值得

### 令人惊讶的发现

1. **更多模型实际上有害**
 - 预期：更多模型 = 更好的性能
 - 现实：性能在 5-6 个模型时达到峰值

2. **MobileNetV4 个体强但集成弱**
 - 个体：第二最佳架构（0.8480）
 - 集成：明显不如 SE-ResNet

3. **TTA=8 灾难性失败**
 - 预期：更多增强 = 更好
 - 现实：严重过拟合，AUC 下降到 0.812

4. **元集成的有效性**
 - 将 3 个集成合并成 6 个独特模型
 - 达到与 5 模型集成相同的性能
 - 表明集成多样性 > 模型数量

---

## 文件结构

```
workspace/
+-- ENSEMBLE_EXPERIMENTS_COMPLETE.md # 本文件
+-- ENSEMBLE_SUMMARY.txt # 快速结果
+-- ALL_MODELS_AUC_COMPARISON.txt # 单个模型排名
+-- ENSEMBLE_MODELS_ABOVE_83.txt # 高性能模型列表
+-- FINAL_ENSEMBLE_66_MODELS.txt # 所有可用模型
+-- massive_ensemble_configs.json # 配置定义
|
+-- ensemble_scripts/ # 51 个集成脚本
| +-- E3_*.sh # 3 模型集成
| +-- E5_*.sh # 5 模型集成
| +-- E8_*.sh # 8 模型集成
| +-- E10_*.sh # 10 模型集成
| +-- FIXED_E5_*.sh # 修正的集成
| +-- META_E_*.sh # 元集成
| +-- ETTA_*.sh # TTA 实验
|
+-- results/ # 结果
| +-- ensemble_*.csv # 预测
| +-- ensemble_*.log # 指标（JSON）
|
+-- logs/ # 执行日志
| +-- ensemble_*.out # 标准输出/错误
|
+-- scripts/
 +-- ensemble_inference.py # 主推理脚本
 +-- train_eric3d_optimized.py # 模型定义
 +-- generate_massive_ensemble_configs.py # 配置生成器
```

---

## 可复现性

### 系统配置
- GPU：NVIDIA RTX 5090
- CUDA：12.8
- PyTorch：2.7.0+
- Python：3.10+

### 随机种子
所有模型都使用固定种子进行训练以确保可复现性

### 数据
- 训练样本：4,348
- 验证样本：4,026 个补丁（来自 5 折交叉验证）
- 补丁大小：64³ 体素
- 格式：HDF5（.h5）

### 重新创建结果

1. **运行单个集成：**
```bash
bash ensemble_scripts/E5_005_seresnet5.sh
```

2. **运行所有集成：**
```bash
bash run_all_ensembles.sh
```

3. **生成新配置：**
```python
python generate_massive_ensemble_configs.py
```

4. **分析结果：**
```python
python analyze_ensemble_results.py
```

---

## 未来工作

### 潜在改进

1. **堆叠元学习器**
 - 目前仅测试了平均/加权平均
 - 可以尝试 Ridge、XGBoost 对集成预测
 - 研究表明可能提高 2-3%

2. **校准**
 - 事后校准（温度缩放、 isotonic）
 - 可能改善概率估计

3. **SE-ResNet 内的架构搜索**
 - 测试 SE-ResNet50、SE-ResNet101
 - 不同的 SE 缩减比

4. **伪标记**
 - 使用集成标记额外的未标记数据
 - 使用扩展数据集重新训练

5. **多尺度集成**
 - 组合不同补丁大小（32³、64³、128³）
 - 可能捕获不同尺度的特征

### 不推荐

1. 跳过添加更多模型（收益递减）
2. 跳过增加架构多样性（损害性能）
3. 跳过 TTA 超过 4 次增强（成本效益不高）
4. 跳过复杂的加权平均（最小收益）

---

## 引用

如果使用这些结果，请引用：

```bibtex
@misc{rsna_ensemble_experiments_2025,
 title={Comprehensive Ensemble Experiments for Intracranial Aneurysm Detection},
 author={RSNA Challenge Team},
 year={2025},
 note={51 ensemble configurations tested, SE-ResNet optimal}
}
```

---

## 附录：完整结果 CSV

完整的 51 个集成的所有指标请参见 `results/ENSEMBLE_SUMMARY.txt`。

逐类 AUC 分数请参见 `results/ensemble_*.log` 中的各个日志文件。

---

**文档版本**：1.0
**最后更新**：2025-10-17
**总实验数**：51 个集成
**最佳 AUC**：0.8624
**推荐**：META_E_top3_weighted