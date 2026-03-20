Brench, 以下内容将该方案整理为 **科研报告或论文式描述**，并对涉及的重要技术术语与评价指标进行简要解释，使整体表述更加规范、系统和完整。

---

# 一、研究背景与总体思路

本研究基于 **RSNA Intracranial Aneurysm Detection Challenge** 数据集开展。该挑战赛的目标是利用深度学习方法，在多模态脑部医学影像（如 CTA、MRA 和 MRI）中自动检测并定位 **颅内动脉瘤（Intracranial Aneurysm）**。参赛者需要预测动脉瘤在 **13 个解剖血管位置**中的存在概率。该数据集由来自 **18 个医疗机构的影像数据组成，并由 60 余名放射科专家完成标注**，具有较高的临床真实性和复杂性。([RSNA][1])

在竞赛评估中，模型性能主要通过 **加权 AUC-ROC（Weighted Area Under the Receiver Operating Characteristic Curve）** 进行衡量，该指标通过对不同类别的 ROC 曲线面积进行加权平均，从而评价模型区分阳性样本与阴性样本的能力。其中整体动脉瘤检测的权重更高，以突出临床检测的重要性。([kaggle.com][2])

针对该任务，本研究提出了一种 **基于分割（Segmentation）与分类（Classification）相结合的多阶段检测框架（multi-stage pipeline）**。整体思路为：

1. 先利用血管分割模型识别血管区域，并生成候选区域（ROI，Region of Interest）。
2. 对每个候选区域进行动脉瘤分类预测。
3. 将 ROI 级别预测结果聚合为病例（Series）级别结果。

该策略能够显著减少无关区域干扰，并提高模型对微小动脉瘤结构的检测能力。

---

# 二、整体模型流程（Pipeline Overview）

整个方法由三个主要阶段组成：

**Stage 1：血管分割与 ROI 提取（Vessel Segmentation and ROI Extraction）**
利用 2.5D 分割模型预测血管掩膜（Vessel Mask），并根据血管类别生成候选检测区域。

**Stage 2：ROI 分类（ROI Classification）**
针对每个 ROI 区域构建分类模型，预测该区域存在动脉瘤的概率。

**Stage 3：ROI 评分与病例级聚合（ROI Scoring and Series-level Aggregation）**
将 ROI 级预测结果整合为病例级预测，并生成最终提交结果。

整体流程如下：

```
医学影像 (DICOM)
      │
      ▼
Stage 1
血管分割 + ROI生成
      │
      ▼
Stage 2
ROI分类 (动脉瘤概率)
      │
      ▼
Stage 3
ROI评分与病例级聚合
      │
      ▼
最终预测结果
```

---

# 三、Stage 1：血管分割模型

## 1 模型结构

血管分割阶段采用 **U-Net 架构**，并使用 **EfficientNet-B0** 作为特征提取骨干网络（backbone）。

**U-Net**
一种广泛用于医学影像分割的卷积神经网络，通过编码器—解码器结构实现像素级预测。

**EfficientNet**
一种高效的卷积神经网络结构，通过复合缩放（compound scaling）在准确率与计算效率之间取得良好平衡。

模型通过 **slice-level segmentation（切片级分割）** 对医学影像中的血管区域进行预测。

---

## 2 输入数据处理

医学影像在输入模型之前进行了多种预处理：

### （1）CT Windowing

CT 影像采用 **窗口化（windowing）** 处理，用于增强特定组织结构的对比度。

CT 原始像素值为 **Hounsfield Unit (HU)**，不同组织具有不同 HU 范围，通过窗口化可突出血管结构。

---

### （2）MRI Percentile Normalization

MRI 图像采用 **百分位归一化（percentile normalization）**，即将像素值映射到指定百分位范围内，以减小不同扫描设备之间的强度差异。

---

### （3）数据增强（Data Augmentation）

为了提升模型泛化能力，训练过程中采用基本的数据增强方法，例如：

* 随机旋转
* 随机翻转
* 随机裁剪

这些方法可增加训练数据的多样性，降低过拟合风险。

---

## 3 输入尺寸

模型输入尺寸为：

```
(C, H, W) = (3, 512, 512)
```

其中：

* **C（Channel）**：输入通道数
* **H / W**：图像高度和宽度

研究发现：

**3 通道输入（multi-slice input）比单通道输入效果更好**，因为它可以包含邻近切片的信息，从而提供更多空间上下文。

---

## 4 类别不平衡问题

医学检测任务通常存在严重的 **类别不平衡（Class Imbalance）**：

* 阴性样本（无动脉瘤）数量远多于阳性样本。

为解决这一问题，本研究采用：

**Weighted Random Sampler**

即对不同类别样本分配不同采样权重，使训练数据中阳性样本比例提高，从而避免模型偏向预测阴性。

---

## 5 分割性能指标

分割模型性能通过 **Dice Score** 进行评估。

### Dice Similarity Coefficient

Dice 系数用于衡量预测分割与真实标注之间的重叠程度：

[
Dice = \frac{2|A∩B|}{|A| + |B|}
]

其中：

* A 为预测区域
* B 为真实标注区域

取值范围：

```
0 → 无重叠
1 → 完全一致
```

本研究分割模型的 **Dice Score ≈ 0.66**。

---

# 四、ROI 提取策略

在完成血管分割后，需要从分割结果中提取 **ROI（Region of Interest）候选区域**。

## ROI 生成方法

每个 ROI 的生成步骤如下：

1. 对分割掩膜计算 **最小外接矩形（bounding box）**
2. 在 bounding box 周围增加一定 **margin（边界扩展）**
3. 得到候选检测区域

当一个切片中同一血管类别存在多个分离区域时：

```
class_i_object_j
```

即每个对象生成独立 ROI。

---

## 正样本 ROI 标注

对于 **含动脉瘤的切片（positive slice）**：

若 ROI 包含动脉瘤坐标，则标记为 **positive ROI**。

该方法的 **hit rate（命中率）约为 94%**，说明 ROI 能覆盖绝大多数动脉瘤。

---

## 负样本 ROI 生成

负样本 ROI 主要来自：

1. 完全阴性的病例
2. 阳性病例中 **距离动脉瘤超过 40 mm 的区域**

这样可以减少模型学习到错误的特征。

---

# 五、Stage 2：ROI 分类模型

在 ROI 级别进行动脉瘤分类预测。

## 1 输入构建（2.5D 输入）

ROI 分类采用 **2.5D 输入方式**：

即将

```
当前切片
+
前一切片
+
后一切片
```

叠加为 3 通道输入。

这种方法能够：

* 保留 **3D空间信息**
* 同时避免 **3D CNN 的高计算成本**

---

## 2 输入尺寸处理

ROI 图像经过以下步骤处理：

1. 按比例缩放（保持长宽比）
2. 将最长边缩放到目标尺寸
3. 对剩余区域进行 padding

最终输入尺寸：

```
(C, H, W) = (3, 224, 224)
```

---

## 3 分类模型结构

实验评估了多种 backbone：

* EfficientNet-B0
* EfficientNet-V2-S

这些网络能够提取高层语义特征，从而识别动脉瘤结构。

---

## 4 分类指标

分类模型通过 **ROC-AUC** 进行评估。

### ROC 曲线

ROC（Receiver Operating Characteristic）曲线表示：

```
True Positive Rate
vs
False Positive Rate
```

### AUC

AUC（Area Under Curve）表示 ROC 曲线下的面积。

取值范围：

```
0.5  → 随机分类
1.0  → 完美分类
```

本研究 ROI 级分类模型：

```
ROI AUC ≈ 0.94
```

说明模型具有较强的区分能力。

---

# 六、Stage 3：ROI 评分与病例级聚合

ROI 分类得到的概率需要进一步转换为 **血管类别预测结果**。

## 1 ROI Score 计算

每个 ROI 的动脉瘤概率：

```
prob
```

来自 ROI 分类模型。

同时分割模型提供：

```
class_scores
```

表示 ROI 属于某血管类别的概率。

最终每个类别的得分：

```
class_score_i = prob × class_scores_i
```

---

## 2 二分类动脉瘤评分

整体动脉瘤检测分数直接使用：

```
prob
```

无需额外变换。

---

## 3 病例级聚合

一个病例中通常存在多个 ROI。

因此需要将 ROI 级预测整合为 **Series-level prediction**。

本研究采用：

**Top-k Mean Aggregation**

步骤：

1. 选择概率最高的 k 个 ROI
2. 计算其平均值

该方法能够减少噪声 ROI 的影响。

---

# 七、模型推理阶段（Inference）

在测试阶段，对 DICOM 影像进行以下处理：

1. 将影像对齐至 **LPS 坐标系（Left-Posterior-Superior）**
2. 仅使用 **Z 轴最后 150 mm 的区域**

原因：

该区域基本对应 **脑部区域**。

此外，为控制计算成本：

```
max_slice = 200
```

即每个病例最多处理 200 张切片。

---

# 八、最终结果

模型最终取得：

```
Local Validation AUC ≈ 0.81
Public Leaderboard Score ≈ 0.81
```

说明该方法在整体检测任务中具有较好的性能。

---

# 九、方法优势

本研究方法的有效之处主要体现在：

1. **ROI 提取策略有效减少背景干扰**
2. **ROI 级分类具有较高 AUC（0.94）**
3. **血管类别概率与 ROI 分类概率结合合理**

这些策略共同提升了检测效果。

---

# 十、方法不足与改进方向

实验中仍存在一些问题：

### 1 Series-level 聚合性能下降

在 ROI → Series 聚合过程中：

```
AUC 出现下降
```

说明部分 ROI 预测存在噪声。

---

### 2 假阳性问题

多种后处理方法尝试去除 **False Positive ROI**，但效果有限。

---

### 3 空间信息利用不足

当前方法主要基于 **2.5D CNN**。

未来可尝试：

* **2.5D CNN + LSTM**
* **3D CNN**

以捕获跨切片空间关系。

---

### 4 MRI T2 模态性能较低

MRI T2 图像在训练中表现较差。

即使训练专门模型：

```
性能仍未明显提升
```

说明不同模态之间存在较大分布差异。

---

# 十一、方法总结

本研究提出了一种 **基于分割引导 ROI 分类的两阶段检测框架**，通过血管分割生成候选区域，并在 ROI 级别进行动脉瘤分类，从而提高检测效率与准确率。

实验结果表明，该方法能够在复杂的多模态医学影像数据中实现较好的动脉瘤检测性能，并在 Kaggle 公榜上取得约 **0.81 的 Weighted AUC**。

整体来看，该框架体现了当前医学影像检测任务中常见的 **Segmentation → ROI → Classification → Aggregation** 的技术路线。

---

如果你愿意，我可以 **再帮你把这套方法改写成“科研面试讲解版本”**（约 2–3 分钟口述版本），这是导师最容易听懂、也最容易追问的表达方式。

[1]: https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge?utm_source=chatgpt.com "RSNA Intracranial Aneurysm Detection AI Challenge (2025)"
[2]: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection?utm_source=chatgpt.com "RSNA Intracranial Aneurysm Detection"
