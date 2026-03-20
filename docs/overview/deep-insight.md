# RSNA 2025 颅内动脉瘤检测

用于从 CT 血管造影扫描中检测和定位颅内动脉瘤的综合 3D 深度学习解决方案。本仓库记录了完整的研究流程，从数据预处理到集成优化。

**竞赛**：[RSNA 2025 颅内动脉瘤检测](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

## 关键结果

- **最佳单模型**：SE-ResNet18 Stable（AUC：0.8585）
- **最佳集成**：META_E_top3_weighted（AUC：0.8624）
- **训练的总模型数**：105 个模型（21 种架构 × 5 折交叉验证）
- **测试的集成配置**：51 种
- **关键发现**：在有限数据上，较小的模型优于较大的模型

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/BrenchCC/RSNA-Intracranial-Aneurysm-Detection-Kaggle-Learning.git
cd RSNA-Intracranial-Aneurysm-Detection-Kaggle-Learning

# 设置环境
conda create -n rsna_kaggle python=3.11
conda activate rsna_kaggle
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
pip install -r requirements.txt

# 运行完整流程
bash scripts/07_run_full_pipeline.sh
```

## 项目概述

### 问题陈述

使用深度学习检测和定位 3D CT 血管造影扫描中 13 个解剖位置的颅内动脉瘤。这是一个多标签分类问题，包含 14 个二元目标：

1. 左侧床突下颈内动脉
2. 右侧床突下颈内动脉
3. 左侧床突上颈内动脉
4. 右侧床突上颈内动脉
5. 左侧大脑中动脉
6. 右侧大脑中动脉
7. 前交通动脉
8. 左侧大脑前动脉
9. 右侧大脑前动脉
10. 左侧后交通动脉
11. 右侧后交通动脉
12. 基底动脉尖
13. 其他后循环
14. **存在动脉瘤**（全局标志）

### 数据集统计

- **总扫描数**：4,348 个 CT 血管造影体积
- **训练集**：每折 3,478 个样本（5 折分层交叉验证）
- **验证集**：每折 870 个样本
- **补丁大小**：64x64x64 体素（0.5mm 间距下的 32mm³）
- **阳性率**：42.8%（存在动脉瘤）
- **类别不平衡**：各解剖位置间高度不平衡

### 方法

**1. 数据预处理**
- DICOM 转换为 NIfTI 格式，保留空间元数据
- HU 窗宽窗位 [-100, 300] 用于 CTA 可视化
- ROI 提取：以脑组织为中心的 64³ 补丁
- Z-分数标准化

**2. 模型架构**
- 3D 卷积神经网络（CNN）
- 重点关注 SE-ResNet 家族（Squeeze-and-Excitation 块）
- 在医学影像数据上从头训练
- 21 种架构，5 折交叉验证

**3. 训练策略**
- 分层 5 折交叉验证
- Adam 优化器（最佳 lr=0.0005）
- 余弦退火学习率调度
- 带类别权重的 BCEWithLogitsLoss
- 早停（patience=15）
- FP16 混合精度训练

**4. 集成优化**
- 测试 51 种集成配置
- 简单平均最优
- 5-6 个模型达到最佳效果
- 纯 SE-ResNet 集成最佳

## 结果摘要

### 排名前 10 的单个模型

| 排名 | 模型 | AUC | 批次大小 | 备注 |
|------|-------|-----|------------|-------|
| 1 | **SE-ResNet18 Stable** | **0.8585** | 12 | LR=0.0005, patience=15 |
| 2 | SE-ResNet18 | 0.8551 | 8 | 原始配置 |
| 3 | ConvNeXt-Large 微调 | 0.8540 | 2 | 冻结 → 微调 |
| 4 | SE-ResNet34 | 0.8538 | 8 | |
| 5 | SE-ResNet50 | 0.8528 | 4 | |
| 6 | DenseNet-121 | 0.8514 | 8 | 完整训练 |
| 7 | DenseNet-121 (v2) | 0.8499 | 8 | 重复运行 |
| 8 | ResNet-18 | 0.8498 | 8 | 标准 ResNet |
| 9 | DenseNet-121 (v3) | 0.8494 | 8 | LR=0.0005 |
| 10 | EfficientNet-B0 | 0.8492 | 8 | |

### 排名前 5 的集成模型

| 排名 | 集成模型 | AUC (宏平均) | AUC (动脉瘤) | 模型数量 | 方法 |
|------|----------|-------------|----------------|--------|--------|
| 1 | E5_005_seresnet5 | **0.8624** | 0.8269 | 5 个 SE-ResNet | 简单平均 |
| 2 | META_E_top3_weighted | **0.8624** | 0.8249 | 6 个 SE-ResNet | 加权平均 |
| 3 | E3_004_seresnet_only | 0.8619 | 0.8248 | 3 个 SE-ResNet | 简单平均 |
| 4 | FIXED_E5_003_seresnet_alt1 | 0.8619 | **0.8298** | 5 个 SE-ResNet | 简单平均 |
| 5 | META_E_top3_ensembles | 0.8619 | 0.8259 | 6 个 SE-ResNet | 简单平均 |

### 关键发现

#### 1. 较小的模型优于较大的模型

**最重要的发现：**
- SE-ResNet18 (0.8585) > SE-ResNet34 (0.8538) > SE-ResNet50 (0.8528)
- DenseNet-121 (0.8514) > DenseNet-169 (0.8430)
- EfficientNet-B0 (0.8492) > B2 (0.8472) > B4 (0.8428)
- ResNet-18 (0.8498) > ResNet-34 (0.8365)

**原因**：~4,000 个训练样本不足以训练大型模型 → 过拟合

#### 2. SE 块对医学影像至关重要

- SE-ResNet18 (0.8585) 与 ResNet-18 (0.8498) 相比 = **+8.7% 改进**
- Squeeze-and-Excitation 注意力机制增强特征学习
- 所有排名前 10 的集成模型均只使用 SE-ResNet

#### 3. 集成的最佳数量：5-6 个模型

| 模型数量 | 最佳 AUC | 性能 |
|--------|----------|-------------|
| 3 | 0.8619 | 最佳的 99.94% |
| 5-6 | **0.8624** | **最优** |
| 10 | 0.8618 | 收益递减 |
| 45 | 0.8582 | -0.5%（更差） |
| 65 | 0.8582 | -0.5%（更差） |

#### 4. 简单平均 > 加权平均

- 加权平均：<0.05% 改进
- 不值得增加复杂性
- 例外：元集成从加权中获益

#### 5. 测试时增强（TTA）不值得

- TTA=4：+0.06% 改进（边际）
- TTA=8：-4.0% 退化（灾难性）
- 推理慢 4-8 倍
- **建议**：生产环境中跳过 TTA

## 仓库结构

```
rsna_github/
+-- README.md # 本文件
+-- PROGRESS_STATUS.md # 开发跟踪
+-- requirements.txt # Python 依赖
+-- .gitignore # Git 忽略规则
|
+-- scripts/ # 顺序流程脚本
| +-- 01_dicom_to_volume.py # DICOM -> NIfTI 转换
| +-- 02_create_roi_patches.py # 提取 64^3 ROI 补丁
| +-- 03_create_cv_splits.py # 生成 5 折 CV 分割
| +-- 04_train_model.py # 主训练脚本（21 种架构）
| +-- 05_ensemble_inference.py # 集成预测
| +-- 06_inference_with_tta.py # 测试时增强
| +-- 07_run_full_pipeline.sh # 完整流程自动化
| |
| +-- utils/ # 模块化工具
| +-- architectures.py # 模型定义
| +-- data_loading.py # 数据集 & 增强
| +-- metrics.py # 训练 & 评估
|
+-- notebooks/ # Jupyter 分析笔记本
| +-- 01_data_exploration_and_eda.ipynb
| +-- 02_preprocessing_and_roi_extraction.ipynb
| +-- 03_architecture_selection_rationale.ipynb
| +-- 04_training_results_analysis.ipynb
| +-- 05_ensemble_strategy_design.ipynb
| +-- 06_ensemble_evaluation_results.ipynb
|
+-- docs/ # 综合文档
| +-- MODEL_DATABASE.md # 所有 105 个模型结果
| +-- ENSEMBLE_RESULTS.md # 51 个集成实验
| +-- GITHUB_ORGANIZATION_PLAN.md # 仓库设计计划
| +-- EXECUTION_SUMMARY.md # 快速参考
| +-- PROGRESS_STATUS.md # 开发状态
|
+-- configs/ # 配置文件
| +-- architectures.yaml # 模型配置
| +-- hyperparameters.yaml # 训练设置
| +-- augmentation.yaml # 数据增强
| +-- ensemble_configs/ # 51 种集成配置
|
+-- launch_scripts/ # 训练自动化
| +-- launch_single_model.sh
| +-- launch_5fold_training.sh
| +-- launch_parallel_training.sh
| +-- monitor_training.sh
|
+-- tests/ # 单元测试
| +-- test_preprocessing.py
| +-- test_models.py
| +-- test_inference.py
|
+-- models/ # 保存的检查点（gitignored）
+-- logs/ # 训练日志（gitignored）
+-- results/ # 预测 & 指标（gitignored）
+-- data/ # 数据集（gitignored）
 +-- raw/ # 原始 DICOM 文件
 +-- volumes_nifti/ # 转换后的 NIfTI 体积
 +-- patches_roi/ # 提取的 64^3 补丁
 +-- cv_splits/ # 5 折 CV 索引
 +-- train_labels_14class.csv # 多标签标注
```

## 流程执行

### 顺序工作流

#### 步骤 1：DICOM 转 NIfTI 转换

```bash
python scripts/01_dicom_to_volume.py \
 --dicom-dir data/raw/train_images \
 --output-dir data/volumes_nifti \
 --num-workers 16
```

**输出**：~4,348 个保留空间元数据的 NIfTI 体积

#### 步骤 2：ROI 补丁提取

```bash
python scripts/02_create_roi_patches.py \
 --nifti-dir data/volumes_nifti \
 --output-dir data/patches_roi \
 --patch-size 64 \
 --window-min -100 \
 --window-max 300
```

**输出**：以脑组织为中心的 64x64x64 体素补丁，Z-分数标准化

#### 步骤 3：交叉验证分割生成

```bash
python scripts/03_create_cv_splits.py \
 --labels-csv data/train_labels_14class.csv \
 --output-dir data/cv_splits \
 --n-folds 5 \
 --seed 42
```

**输出**：分层 5 折分割，确保正负样本平衡

#### 步骤 4：模型训练

```bash
# 单模型训练
python scripts/04_train_model.py \
 --data-dir data/patches_roi \
 --labels-csv data/train_labels_14class.csv \
 --cv-dir data/cv_splits \
 --fold 0 \
 --arch seresnet18_stable \
 --batch-size 12 \
 --lr 0.0005 \
 --epochs 50 \
 --patience 15 \
 --output models/fold0_seresnet18_stable
```

**支持的架构**：
- SE-ResNet 家族：seresnet10, seresnet14, seresnet18, seresnet34, seresnet50
- DenseNet 家族：densenet121, densenet169
- ResNet 家族：resnet18, resnet34, resnet50
- EfficientNet 家族：efficientnet_b0, efficientnet_b2, efficientnet_b3, efficientnet_b4
- MobileNet 家族：mobilenetv3
- Vision Transformers：vit, swin, convnext
- 其他：inception, unet3d

#### 步骤 5：集成推理

```bash
# 生产级集成（6 个模型，加权）
python scripts/05_ensemble_inference.py \
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
 --val-dir data/patches_roi \
 --val-csv data/train_labels_14class.csv \
 --output results/ensemble_predictions.csv
```

#### 步骤 6：测试时增强（可选）

```bash
python scripts/06_inference_with_tta.py \
 --model models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 --arch seresnet18_stable \
 --val-dir data/patches_roi \
 --val-csv data/train_labels_14class.csv \
 --tta-count 4 \
 --output results/tta_predictions.csv
```

**注意**：TTA=4 提供最小的收益（+0.06%），但推理慢 4 倍。生产环境中请跳过。

### 并行训练

在双 GPU 上同时训练多个模型：

```bash
# 启动并行训练（2 GPU）
bash launch_scripts/launch_parallel_training.sh

# 监控训练进度
bash launch_scripts/monitor_training.sh

# 查看特定日志
tail -f logs/train_fold0_seresnet18_stable.log
```

## 硬件要求

### 作者的训练设置

所有 105 个模型在两台工作站上本地训练：

**主工作站**：
- **GPU**：2x NVIDIA RTX 5090（每个 32GB VRAM）
- **CPU**：AMD/Intel 多核（16+ 线程）
- **RAM**：128GB
- **存储**：4TB NVMe SSD（2x 2TB WD Black SN850X RAID 0）
- **操作系统**：Ubuntu 22.04 LTS

**次要 PC**：
- **GPU**：1x NVIDIA RTX 3090 Ti（24GB VRAM）+ 1x RTX 3090（24GB VRAM）
- **总训练 GPU**：4 个

这种多 GPU 设置使不同架构能够在 21 种架构 × 5 折的实验空间中并行训练。

### 最低要求

- **GPU**：NVIDIA GPU 16GB+ VRAM（推荐 RTX 4090 或 RTX 3090）
- **RAM**：64GB
- **存储**：500GB SSD
- **CUDA**：12.1+
- **PyTorch**：2.0+

### 批次大小

| 模型 | 批次大小 | 参数 |
|-------|------------|--------|
| SE-ResNet18 | 12 | ~11M |
| SE-ResNet34 | 8 | ~22M |
| SE-ResNet50 | 4 | ~28M |
| DenseNet-121 | 8 | ~8M |

## 模型架构详情

### SE-ResNet18 Stable（最佳模型）

```
Input: (1, 64, 64, 64) # 单通道 3D CT 补丁
+-- Conv3d(1->64, 7x7x7, stride=2) # 初始特征提取
+-- BatchNorm3d + ReLU + MaxPool3d
|
+-- Layer1: 2 x SE-BasicBlock(64->64) # 空间大小: 16x16x16
+-- Layer2: 2 x SE-BasicBlock(64->128, stride=2) # 8x8x8
+-- Layer3: 2 x SE-BasicBlock(128->256, stride=2) # 4x4x4
+-- Layer4: 2 x SE-BasicBlock(256->512, stride=2) # 2x2x2
|
+-- AdaptiveAvgPool3d -> (512, 1, 1, 1)
+-- Flatten -> (512,)
+-- Linear(512->14) -> 多标签输出
```

**SE-BasicBlock**：
```
Input: x
+-- Conv3d(3x3x3) -> BN -> ReLU -> Conv3d(3x3x3) -> BN
+-- Squeeze-and-Excitation:
| +-- 全局平均池化 -> (C,)
| +-- FC(C->C/16) -> ReLU -> FC(C/16->C) -> Sigmoid
| +-- 通道乘法
+-- 残差连接: x + SE(Conv(x))
```

**参数**：~11M
**VRAM**：~2.5GB（批次大小 12）

## 数据增强

保留解剖学有效性的医学影像专用 3D 增强：

```python
VolumeAugmentation(
 rotation_range=15, # +/-15 度（保留解剖结构）
 flip=True, # 所有轴（对大脑有效）
 zoom_range=(0.9, 1.1), # 保守缩放
 shift_range=0.1, # +/-10% 平移
 brightness_range=0.2, # +/-20% 强度
 contrast_range=0.2 # +/-20% 对比度
)
```

**原理**：
- 旋转限制在 +/-15 度以避免不真实的解剖位置
- 所有翻转均有效（大脑的双侧对称性）
- 保守缩放以保持血管结构尺度
- 强度增强考虑扫描器/协议变化

## 训练配置

### 超参数（优化后）

```yaml
# 模型
architecture: seresnet18_stable
patch_size: 64
num_classes: 14

# 优化
optimizer: Adam
learning_rate: 0.0005 # 最优（0.001 导致不稳定）
weight_decay: 1e-4
lr_scheduler: CosineAnnealingLR
min_lr: 1e-6

# 训练
batch_size: 12 # SE-ResNet18 最优
epochs: 50
patience: 15 # 早停
gradient_clip: 1.0

# 数据
augmentation: true
num_workers: 8
pin_memory: true

# 损失
criterion: BCEWithLogitsLoss
class_weights: computed # 从训练集分布计算
label_smoothing: 0.0

# 精度
mixed_precision: true # FP16 加快训练
```

### 类别权重

从训练集计算以处理类别不平衡：

| 类别 | 阳性率 | 权重 |
|-------|---------------|--------|
| 存在动脉瘤 | 42.8% | 1.34 |
| 左侧床突下 ICA | 8.2% | 11.2 |
| 右侧床突下 ICA | 7.9% | 11.6 |
| 左侧床突上 ICA | 12.4% | 7.1 |
| 右侧床突上 ICA | 11.8% | 7.5 |
| 左侧 MCA | 18.3% | 4.5 |
| 右侧 MCA | 17.6% | 4.7 |
| 前交通动脉 | 9.4% | 9.6 |
| 左侧 ACA | 5.3% | 17.9 |
| 右侧 ACA | 5.1% | 18.6 |
| 左侧 PComm | 6.8% | 13.7 |
| 右侧 PComm | 6.2% | 15.1 |
| 基底动脉尖 | 7.4% | 12.5 |
| 其他后循环 | 1.2% | 82.3 |

## 集成策略

### 推荐的生产集成

#### 1. 最高性能：META_E_top3_weighted

**AUC**：0.8624（并列最佳）

```bash
python scripts/05_ensemble_inference.py \
 --config-id META_E_top3_weighted \
 --method weighted \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 models/fold3_seresnet18_p64/best_model.pth \
 --weights 2.5862 2.5862 1.7243 1.7243 1.7243 0.8619
```

**优点**：最稳健，6 个多样化模型，元集成稳定性
**缺点**：稍慢（6 个模型）

#### 2. 平衡：E5_005_seresnet5

**AUC**：0.8624（并列最佳）

```bash
python scripts/05_ensemble_inference.py \
 --config-id E5_005_seresnet5 \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth
```

**优点**：简单平均，出色性能，5 个模型
**缺点**：无

#### 3. 快速推理：E3_004_seresnet_only

**AUC**：0.8619（最佳的 99.94%）

```bash
python scripts/05_ensemble_inference.py \
 --config-id E3_004_seresnet_only \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth
```

**优点**：仅 3 个模型（2 倍速），均为 fold 0（易于部署）
**缺点**：AUC 稍低（-0.05%）

## 可复现性

### 随机种子

所有实验使用固定随机种子：

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 交叉验证

分层 5 折交叉验证：
- 各折间标签分布平衡（+/-2% 容差）
- 患者级分割（无数据泄漏）
- 种子随机状态确保可复现分割

### 环境

```yaml
python: 3.11
pytorch: 2.7.0+cu128
cuda: 12.8
cudnn: 9.3.0
numpy: 2.2.3
pandas: 2.2.3
scikit-learn: 1.6.1
nibabel: 5.3.2
scipy: 1.15.1
```

完整依赖列表在 `requirements.txt` 中。

## 引用

如果您在研究中使用此代码或发现，请引用：

```bibtex
@misc{dalbey2025rsna_aneurysm,
 title={RSNA 2025 Intracranial Aneurysm Detection: Comprehensive Deep Learning Pipeline},
 author={Dalbey, Glenn},
 year={2025},
 publisher={GitHub},
 url={https://github.com/XxRemsteelexX/RSNA-Intracranial-Aneurysm-Detection-Kaggle},
 note={Best AUC: 0.8624 (ensemble), 0.8585 (single model)}
}
```

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)

## 致谢

- **RSNA** 组织本次挑战赛
- **Kaggle** 主办竞赛
- **PyTorch** 团队提供深度学习框架
- **医学影像社区** 提供 CTA 预处理最佳实践

## 作者说明

这是我的第一次 Kaggle 竞赛。我在本地硬件上训练了 105 个模型并测试了 51 种集成配置：主工作站上的 2x NVIDIA RTX 5090 GPU 和第二台 PC 上的额外 RTX 3090 Ti + RTX 3090。

不幸的是，我没有意识到提交必须在比赛截止日期前 24 小时完成——最后一天只能从之前提交的模型中选择。尽管无法提交最终结果，这项工作代表了对 3D 医学影像的全面探索。

关键发现——较小的模型在有限的医学影像数据集上显著优于较大的模型——来自于系统地测试 SOTA 大型模型会效果最佳的假设，记录它们的失败，并迭代向更简单的解决方案。有时最好的发现来自于不起作用的方法。

## 联系方式

- **作者**：Glenn Dalbey
- **GitHub**：[@XxRemsteelexX](https://github.com/XxRemsteelexX)
- **电子邮件**：dalbeyglenn@gmail.com

## 文档

详细信息请参阅：

- [MODEL_DATABASE.md](docs/results/MODEL_DATABASE.md) - 所有 105 个模型的完整结果
- [ENSEMBLE_RESULTS.md](docs/results/ENSEMBLE_RESULTS.md) - 所有 51 种集成实验
- [GITHUB_ORGANIZATION_PLAN.md](docs/GITHUB_ORGANIZATION_PLAN.md) - 仓库设计原理
- [PROGRESS_STATUS.md](docs/PROGRESS_STATUS.md) - 开发跟踪

---

**状态**：生产就绪 | **最佳 AUC**：0.8624（集成）| **最后更新**：2025-10-17
