# Docs Index

`docs/` 已按长期维护的方式重构为 4 个编号目录：

- `01-overview/`：背景、当前方法、训练边界、研究笔记、摘要
- `02-setup/`：复现与推理说明
- `03-results/`：单模型与集成实验结论
- `04-tricks/`：关键技巧拆解

## 文档关系图

```mermaid
flowchart TD
    A[docs/README.md<br/>总入口]
    B[01-overview/introduction.md<br/>任务背景]
    C[01-overview/current-method.md<br/>当前方法]
    D[01-overview/current-training.md<br/>训练边界]
    E[01-overview/research-notes.md<br/>后续研究]
    F[02-setup/*.md<br/>复现与推理]
    G[03-results/*.md<br/>实验结果]
    H[04-tricks/*.md<br/>技巧拆解]

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
    A --> H
    B --> C
    C --> D
    C --> F
    D --> G
    G --> E

    click A "./README.md" "打开 docs 索引"
    click B "./01-overview/introduction.md" "打开任务背景"
    click C "./01-overview/current-method.md" "打开当前方法"
    click D "./01-overview/current-training.md" "打开训练边界"
    click E "./01-overview/research-notes.md" "打开后续研究"
    click F "./02-setup/project-setup-cn.md" "打开复现说明"
    click G "./03-results/model-database-cn.md" "打开实验结果"
    click H "./04-tricks/README.md" "打开技巧总览"
```

图中的节点可以直接点击跳转到对应文档。

## 建议阅读顺序

### 第一次进入项目

1. [introduction.md](./01-overview/introduction.md)
2. [guide.md](./01-overview/guide.md)
3. [current-method.md](./01-overview/current-method.md)
4. [current-training.md](./01-overview/current-training.md)

### 准备复现当前方案

1. [current-method.md](./01-overview/current-method.md)
2. [current-training.md](./01-overview/current-training.md)
3. [project-setup-cn.md](./02-setup/project-setup-cn.md)
4. [inference-setup-cn.md](./02-setup/inference-setup-cn.md)

### 准备看实验与下一步优化

1. [model-database-cn.md](./03-results/model-database-cn.md)
2. [ensemble-results-cn.md](./03-results/ensemble-results-cn.md)
3. [research-notes.md](./01-overview/research-notes.md)

## 目录说明

### `01-overview/`

- [introduction.md](./01-overview/introduction.md)
  项目背景、任务定义、数据与评估方式。

- [guide.md](./01-overview/guide.md)
  `01-overview/` 子目录导航，说明每篇文档职责。

- [current-method.md](./01-overview/current-method.md)
  当前采用的方法说明。

- [current-training.md](./01-overview/current-training.md)
  当前方法里哪些模块真的训练、哪些只是规则或聚合。

- [research-notes.md](./01-overview/research-notes.md)
  后续研究文档。

- [summary-cn.md](./01-overview/summary-cn.md)
  中文摘要版总览。

- [summary-en.md](./01-overview/summary-en.md)
  英文摘要版总览。

### `02-setup/`

- [project-setup-cn.md](./02-setup/project-setup-cn.md)
- [project-setup-en.md](./02-setup/project-setup-en.md)
- [inference-setup-cn.md](./02-setup/inference-setup-cn.md)
- [inference-setup-en.md](./02-setup/inference-setup-en.md)

这一组文档只负责“怎么落地当前方案”。

### `03-results/`

- [model-database-cn.md](./03-results/model-database-cn.md)
- [model-database-en.md](./03-results/model-database-en.md)
- [ensemble-results-cn.md](./03-results/ensemble-results-cn.md)
- [ensemble-results-en.md](./03-results/ensemble-results-en.md)

这一组文档只负责“实验结论是什么”。

### `04-tricks/`

- [README.md](./04-tricks/README.md)
- [segmentation-to-detection.md](./04-tricks/segmentation-to-detection.md)
- [vessel-class-roi.md](./04-tricks/vessel-class-roi.md)
- [distance-aware-negative-sampling.md](./04-tricks/distance-aware-negative-sampling.md)
- [topk-mean-aggregation.md](./04-tricks/topk-mean-aggregation.md)
- [input-2p5d.md](./04-tricks/input-2p5d.md)

这一组文档只负责“方法细节拆解”，不重复承担背景和结果总结。
