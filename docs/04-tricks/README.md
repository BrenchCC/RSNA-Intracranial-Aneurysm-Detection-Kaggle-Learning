# 动脉瘤检测 5 个关键 Tricks（总览）

本文将完整方案拆为 5 个“非通用但高收益”的技巧。它们不是常规调参，而是围绕医学影像的结构先验、候选生成与病例级决策设计。

## 技巧关系图

```mermaid
flowchart TD
    A[Trick 1<br/>segmentation-to-detection]
    B[Trick 2<br/>vessel-class-roi]
    C[Trick 3<br/>distance-aware-negative-sampling]
    D[Trick 4<br/>topk-mean-aggregation]
    E[Trick 5<br/>input-2p5d]

    A --> B
    A --> C
    B --> D
    C --> D
    E --> D
    A --> E

    click A "./segmentation-to-detection.md" "打开 Trick 1"
    click B "./vessel-class-roi.md" "打开 Trick 2"
    click C "./distance-aware-negative-sampling.md" "打开 Trick 3"
    click D "./topk-mean-aggregation.md" "打开 Trick 4"
    click E "./input-2p5d.md" "打开 Trick 5"

    style A fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style B fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style C fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style E fill:#FFF3E0,stroke:#FB8C00,stroke-width:1.5px
    style D fill:#F3E5F5,stroke:#8E24AA,stroke-width:1.5px
```

图中的节点可以直接点击跳转到对应文件。

颜色说明：

- 绿色：候选生成与 ROI 构建相关技巧
- 橙色：分类相关技巧
- 紫色：聚合相关技巧

## Tricks 嵌入总体 Pipeline 的位置

```mermaid
flowchart LR
    A[原始体数据]
    B[血管约束候选]
    C[ROI 构建]
    D[ROI 分类]
    E[病例级聚合]
    F[最终预测]

    T1[Trick 1<br/>先分割再检测]
    T2[Trick 2<br/>血管类别 ROI]
    T3[Trick 3<br/>距离约束负样本]
    T4[Trick 4<br/>Top-k 聚合]
    T5[Trick 5<br/>2.5D 输入]

    A --> B --> C --> D --> E --> F

    T1 -.作用于.-> B
    T2 -.作用于.-> C
    T3 -.作用于.-> C
    T5 -.作用于.-> D
    T4 -.作用于.-> E

    click T1 "./segmentation-to-detection.md" "查看 Trick 1"
    click T2 "./vessel-class-roi.md" "查看 Trick 2"
    click T3 "./distance-aware-negative-sampling.md" "查看 Trick 3"
    click T4 "./topk-mean-aggregation.md" "查看 Trick 4"
    click T5 "./input-2p5d.md" "查看 Trick 5"

    style A fill:#E3F2FD,stroke:#1E88E5,stroke-width:1.5px
    style F fill:#E3F2FD,stroke:#1E88E5,stroke-width:1.5px

    style B fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style C fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px

    style D fill:#FFF3E0,stroke:#FB8C00,stroke-width:1.5px

    style E fill:#F3E5F5,stroke:#8E24AA,stroke-width:1.5px

    style T1 fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style T2 fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style T3 fill:#E8F5E9,stroke:#43A047,stroke-width:1.5px
    style T5 fill:#FFF3E0,stroke:#FB8C00,stroke-width:1.5px
    style T4 fill:#F3E5F5,stroke:#8E24AA,stroke-width:1.5px
```

这张图更适合回答一个实际问题：每个 trick 到底插在 pipeline 的哪一段。

颜色说明：

- 蓝色：输入与最终输出
- 绿色：候选生成 / ROI 构建
- 橙色：分类
- 紫色：聚合

## 文档索引
1. [Trick 1: 先分割血管再检测](./segmentation-to-detection.md)
2. [Trick 2: 按血管类别生成 ROI](./vessel-class-roi.md)
3. [Trick 3: 距离约束负样本采样](./distance-aware-negative-sampling.md)
4. [Trick 4: Top-k Mean 聚合](./topk-mean-aggregation.md)
5. [Trick 5: 2.5D 输入](./input-2p5d.md)

## 一句话理解每个 Trick
- Trick 1：先把“全脑搜索”收缩为“血管内搜索”，显著降低背景噪声。
- Trick 2：让 ROI 自带血管身份，显式引入解剖先验，定位更稳定。
- Trick 3：负样本采样加距离约束，减少 near-lesion 混淆噪声。
- Trick 4：病例级结果不看单点最大值，而看 top-k 证据均值，提高鲁棒性。
- Trick 5：用 2.5D 在接近 2D 成本下引入局部 3D 结构信息。

## 组合后的 Pipeline（建议表述）
1. 先做血管分割，限制候选搜索空间。
2. 按血管类别生成 ROI，并做类别感知分类打分。
3. 训练时用距离约束构造更干净的负样本。
4. 推理时对 ROI 分数做血管内 top-k 聚合得到病例/血管级预测。
5. ROI 输入采用 2.5D，补足单切片上下文不足。

## 面试/复盘可重点讲的两个点
- `Segmentation -> ROI Detection`：这是误检控制和效率提升的核心杠杆。
- `Vessel-Class Prior + Top-k Aggregation`：一个负责“结构正确”，一个负责“决策稳健”。
