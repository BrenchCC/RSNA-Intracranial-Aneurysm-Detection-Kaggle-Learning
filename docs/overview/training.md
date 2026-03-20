Brench, 简单直接回答：

**这个方案实际训练了两个模型**，而不是三个。
Pipeline 虽然有 **Stage1 / Stage2 / Stage3**，但只有前两部分涉及模型训练，第三部分只是计算与聚合规则。

下面逐一说明。

---

# 一、真正训练的模型

## 1️⃣ Stage1：血管分割模型（Segmentation Model）

训练了一个 **医学图像分割模型**。

模型结构：

* **U-Net**
* backbone：**EfficientNet-B0**

任务：

```text
输入：CT / MRI slice
输出：血管区域 mask
```

也就是：

```text
image → vessel segmentation mask
```

作用：

* 找到 **血管位置**
* 为后续检测提供 **ROI候选区域**

为什么要做这个模型？

因为：

**动脉瘤只会出现在血管上。**

先分割血管可以：

* 减少搜索空间
* 降低误检

分割模型评价指标：

**Dice Score ≈ 0.66**

Dice 是分割任务常用指标：

```text
Dice = 2 * overlap / (prediction + ground truth)
```

衡量预测区域和真实标注的重叠程度。

---

## 2️⃣ Stage2：ROI 分类模型（Classification Model）

第二个训练的是 **分类模型**。

任务：

```text
输入：ROI 图像
输出：该区域是否存在动脉瘤
```

具体输入：

* 当前 slice
* 前一个 slice
* 后一个 slice

形成：

```text
3 channel image
```

这就是所谓：

**2.5D CNN**

解释：

| 方法       | 含义     |
| -------- | ------ |
| 2D CNN   | 单张图像   |
| 3D CNN   | 整个体数据  |
| 2.5D CNN | 相邻切片拼接 |

2.5D 的优点：

* 利用部分 3D 信息
* 计算量远低于 3D CNN

分类模型 backbone：

* EfficientNet-B0
* EfficientNet-V2-S

分类指标：

**ROI-level AUC ≈ 0.94**

AUC 的含义：

* 衡量分类模型区分能力
* 0.5 = 随机
* 1 = 完美

0.94 说明 **ROI 分类能力非常强**。

---

# 二、没有训练模型的部分

## Stage3：评分与聚合（没有训练）

这一部分只是 **规则计算**。

主要做三件事。

---

### 1 ROI probability

分类模型输出：

```text
prob = P(aneurysm | ROI)
```

---

### 2 血管类别概率

分割模型会预测 ROI 属于哪种血管：

```text
class_scores
```

例如：

| 血管       | 概率  |
| -------- | --- |
| ICA_left | 0.7 |
| MCA_left | 0.2 |
| ACA      | 0.1 |

---

### 3 最终类别得分

计算公式：

```text
class_score = prob × class_score
```

得到 **13个血管类别概率**。

竞赛要求预测：

```text
13 vascular locations
```

例如：

* ICA
* MCA
* ACA
* Basilar artery

参赛者需要预测动脉瘤是否出现在这些血管位置。 ([rsna.org][1])

---

### 4 Series-level aggregation

一个病例通常有：

```text
几十到几百个 ROI
```

需要汇总为：

```text
1 个病例预测
```

采用方法：

**Top-k mean**

步骤：

1. 找概率最高的 k 个 ROI
2. 取平均

例如：

```text
top 5 ROI prob
→ mean
```

得到病例预测结果。

---

# 三、整体总结（面试回答版）

如果导师问：

**“你这个方案训练了什么模型？”**

可以这样回答：

> 该方法主要训练了两个深度学习模型。
> 第一阶段训练一个基于 U-Net 的血管分割模型，用于从脑部医学影像中预测血管区域，并生成候选 ROI。
> 第二阶段训练一个 ROI 级别的分类模型（EfficientNet backbone），对每个候选区域预测动脉瘤存在概率。
> 第三阶段并未进行模型训练，而是通过规则方法将 ROI 预测结果与血管类别概率结合，并通过 top-k mean 的方式进行病例级结果聚合。

一句话总结：

```text
训练了两个模型：
1 Segmentation
2 Classification
```
