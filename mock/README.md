# Mock 全局流程结构分析（小白版）

这份文档只做一件事：
- 让你在没有真实 DICOM、没有真实模型的情况下，完整看懂 `mock/` 四阶段 pipeline 每一步到底吃什么、吐什么。

你可以把它当成“流程拆解 + 数据格式说明书”。

---

## 0. 先建立一个直觉：这条链路在做什么？

输入是一个 3D 体数据（`volume.npy`）。

然后依次经过：
1. Stage1：预处理 + 生成血管先验图。
2. Stage2：在血管先验上找候选点，裁 ROI，做距离规则过滤。
3. Stage3：把 ROI 做 2.5D 打分，得到每个 ROI 的概率。
4. Stage4：把一堆 ROI 概率聚合成最终 14 维病例级结果。

最终输出是：
- `case_pred.json`（14 维概率）
- `submission_like.csv`（仿提交格式）

---

## 1. 全局目录结构（你会看到什么文件）

```text
mock/
|-- create_mock_input.py
|-- stage1_preprocess.py
|-- stage2_candidates.py
|-- stage3_roi_classifier.py
|-- stage4_aggregate.py
|-- run_all.py
|-- constants.py
|-- visualize.py
|-- input/
|   `-- <case_id>/
|       |-- volume.npy
|       `-- case_meta.json
`-- output/
    `-- <case_id>/
        |-- stage1/
        |   |-- artifacts/
        |   `-- figs/debug + figs/report
        |-- stage2/
        |   |-- artifacts/
        |   `-- figs/debug + figs/report
        |-- stage3/
        |   |-- artifacts/
        |   `-- figs/debug + figs/report
        `-- stage4/
            |-- artifacts/
            `-- figs/debug + figs/report
```

---

## 2. 一键跑全流程

```bash
python mock/run_all.py \
  --case-id case_001 \
  --input-dir mock/input \
  --output-dir mock/output \
  --seed 42 \
  --num-roi 64 \
  --k-top 5 \
  --num-slices 8
```

说明：
- 如果 `mock/input/case_001` 不存在，会自动生成 mock 输入。
- `--num-slices` 必须在 `[6, 10]`。

---

## 3. 每个阶段的输入输出（含矩阵/张量形式）

下面是最重要的部分。

### Stage0（可选）：`create_mock_input.py`

作用：生成一个可测试的输入病例。

输入：
- 命令行参数 `--case-id --seed --shape`。

输出：
- `volume.npy`
- `case_meta.json`

矩阵/张量形式：
- `volume.npy`: `V ∈ R^(Z×Y×X)`
- 默认形状：`(64, 128, 128)`，dtype = `float32`

`case_meta.json` 关键字段：
- `shape_zyx`: `[Z, Y, X]`
- `spacing_zyx_mm`: `[sz, sy, sx]`

---

### Stage1：`stage1_preprocess.py`

作用：把原体数据标准化，并构造血管先验概率图（Trick1）。

输入文件：
- `mock/input/<case_id>/volume.npy`
- `mock/input/<case_id>/case_meta.json`

输出文件：
- `mock/output/<case_id>/stage1/artifacts/volume_norm.npy`
- `mock/output/<case_id>/stage1/artifacts/vessel_prior.npy`
- `mock/output/<case_id>/stage1/artifacts/stage1_meta.json`

矩阵/张量形式：
- 原始体数据：`V ∈ R^(Z×Y×X)`
- 标准化体数据：`V_norm ∈ R^(Z×Y×X)`
- 血管先验图：`P_vessel ∈ [0,1]^(Z×Y×X)`

你可以理解为：
- `V_norm[z,y,x]` = 这个体素标准化后的强度
- `P_vessel[z,y,x]` = 这个体素“像血管”的概率

可视化：
- 原图中间层
- 标准化中间层
- 血管先验叠加图

---

### Stage2：`stage2_candidates.py`

作用：
- 在 `P_vessel` 上采样候选中心点。
- 给每个候选绑定血管类别（Trick2）。
- 计算到伪病灶距离并按阈值过滤（Trick3 模拟）。
- 按候选中心裁 3D ROI patch。

输入文件：
- `mock/output/<case_id>/stage1/artifacts/vessel_prior.npy`
- `mock/output/<case_id>/stage1/artifacts/volume_norm.npy`
- `mock/output/<case_id>/stage1/artifacts/stage1_meta.json`

输出文件：
- `mock/output/<case_id>/stage2/artifacts/candidates.json`
- `mock/output/<case_id>/stage2/artifacts/roi_patches.npy`
- `mock/output/<case_id>/stage2/artifacts/stage2_meta.json`

矩阵/张量形式：
- ROI patch 批量：`X_roi ∈ R^(N×D×H×W)`
- 当前默认：`N = num_roi`，`D = H = W = patch_size`（默认 `24`）
- 例如：`(64, 24, 24, 24)`

`candidates.json` 核心字段：
- `case_id`
- `candidates: [ ... ]`
- 每个候选：
  - `id`: 整数编号
  - `center_xyz`: `[z, y, x]`
  - `vessel_class`: 血管类别字符串
  - `prior_score`: 来自 `P_vessel` 的先验分
  - `dist_to_pseudo_lesion_mm`: 到伪病灶最近距离（毫米）
  - `keep_by_distance_rule`: 是否通过距离规则过滤

可视化：
- 候选散点图
- 类别计数图
- 距离过滤前后计数图
- 距离分布直方图
- ROI 裁剪网格图

---

### Stage3：`stage3_roi_classifier.py`

作用：
- 对每个 3D ROI 取 2.5D 多切片（Trick5）。
- 生成每个 ROI 的 `p_a`, `p_c`, `fused_score`。

输入文件：
- `mock/output/<case_id>/stage2/artifacts/roi_patches.npy`
- `mock/output/<case_id>/stage2/artifacts/candidates.json`

输出文件：
- `mock/output/<case_id>/stage3/artifacts/roi_scores.json`
- `mock/output/<case_id>/stage3/artifacts/stage3_meta.json`

矩阵/张量形式：
- 输入 patch 批量：`X_roi ∈ R^(N×D×H×W)`
- 每个 ROI 的 2.5D 抽取：`X_2.5d(i) ∈ R^(S×H×W)`
- `S = num_slices`，范围 `[6,10]`，默认 `8`

概率输出形式：
- `p_a ∈ [0,1]`: ROI 动脉瘤概率
- `p_c ∈ [0,1]`: ROI 类别一致性概率
- `fused_score = p_a * p_c`

`roi_scores.json` 核心字段：
- `case_id`
- `roi_scores: [{id, vessel_class, p_a, p_c, fused_score}, ...]`

可视化：
- ROI 分数条形图（Top）
- Top 证据 patch 拼图

---

### Stage4：`stage4_aggregate.py`

作用：
- 按 `vessel_class` 分组 ROI 分数。
- 每组做 `top-k mean`（Trick4）。
- 生成病例级 14 维预测。

输入文件：
- `mock/output/<case_id>/stage3/artifacts/roi_scores.json`

输出文件：
- `mock/output/<case_id>/stage4/artifacts/case_pred.json`
- `mock/output/<case_id>/stage4/artifacts/submission_like.csv`
- `mock/output/<case_id>/stage4/artifacts/stage4_meta.json`

向量形式：
- 对每个血管类别 `c`，有 ROI 分数集合 `P_c = {p1, p2, ...}`
- 聚合：`S_c = mean(top_k(P_c))`
- 最终 14 维：
  - 前 13 维：位置概率
  - 第 14 维：`Aneurysm Present = max(前13维)`

`case_pred.json` 核心字段：
- `case_id`
- `method`
- `k`
- `pred_14`（固定 14 个字段名，含 `Aneurysm Present`）

可视化：
- 14 维概率柱状图
- `max` vs `top-k mean` 对比图

---

## 4. 全流程数据形状总表（一眼看懂）

| 阶段 | 文件 | 数学形式 | 示例 shape | 含义 |
|---|---|---|---|---|
| 输入 | `volume.npy` | `V ∈ R^(Z×Y×X)` | `(64,128,128)` | 原始体数据 |
| Stage1 输出 | `volume_norm.npy` | `V_norm ∈ R^(Z×Y×X)` | `(64,128,128)` | 标准化体数据 |
| Stage1 输出 | `vessel_prior.npy` | `P_vessel ∈ [0,1]^(Z×Y×X)` | `(64,128,128)` | 血管先验概率图 |
| Stage2 输出 | `roi_patches.npy` | `X_roi ∈ R^(N×D×H×W)` | `(64,24,24,24)` | 候选 ROI 批量 |
| Stage3 中间 | 2.5D 切片 | `X_2.5d(i) ∈ R^(S×H×W)` | `(8,24,24)` | 单 ROI 的 2.5D 输入 |
| Stage3 输出 | `roi_scores.json` | `N` 个标量集合 | `N=64` | 每个 ROI 的概率 |
| Stage4 输出 | `pred_14` | `y ∈ [0,1]^14` | `14` | 病例级最终概率 |

---

## 5. 小白最常见问题（FAQ）

### Q1：为什么有 `debug/` 和 `report/` 两套图？
- `debug/`：排查问题，信息更密。
- `report/`：展示汇报，更简洁。

### Q2：为什么 Stage2 有“距离过滤”，但这是推理流程？
- 这里是 Trick3 的“模拟展示版”，用于可视化讲清规则，不是声称真实训练收益。

### Q3：`num_slices` 为什么限制 6-10？
- 这是当前 mock 演示的约束，保证 2.5D 展示既有上下文又不会太重。

### Q4：我只想看最终结果看哪个文件？
- 看：`mock/output/<case_id>/stage4/artifacts/case_pred.json`

### Q5：我想检查每一步是否合理？
- 按顺序看：
  - Stage1 overlay 图
  - Stage2 候选散点 + 距离过滤图
  - Stage3 Top 证据图
  - Stage4 14 维柱状图

---

## 6. 与真实 pipeline 的边界

- 这是 mock 教学链路，不做真实 DICOM 解码。
- 不加载真实深度模型权重。
- 重点是“流程和接口可解释”，不是“医学准确率”。

如果以后要接入真实系统，通常替换：
- Stage1 的 `vessel_prior` 生成方式
- Stage3 的打分方式（真实模型推理）
- 其余 I/O 契约基本可复用。

---

## 7. 分阶段命令（教学用）

```bash
python mock/create_mock_input.py --input-dir mock/input --case-id case_001 --seed 42
python mock/stage1_preprocess.py --case-id case_001 --input-dir mock/input --output-dir mock/output --seed 42
python mock/stage2_candidates.py --case-id case_001 --output-dir mock/output --seed 42 --num-roi 64
python mock/stage3_roi_classifier.py --case-id case_001 --output-dir mock/output --seed 42 --num-slices 8
python mock/stage4_aggregate.py --case-id case_001 --output-dir mock/output --k-top 5
```

你可以每跑一步就打开对应 `stageX/artifacts` 和 `stageX/figs`，边跑边理解。

---

## 8. 看图顺序（从输入到最终输出）

这一节给你一个固定阅读顺序。你按这个顺序看图，最容易快速理解全流程。

### 第 1 步：先看输入图（确认原始体数据是否正常）
- `mock/output/<case_id>/input_preview/figs/report/input_mid_slice.png`
- `mock/output/<case_id>/input_preview/figs/report/input_slice_grid.png`

你要看什么：
- 切片是否连续。
- 是否有明显异常空白层或破碎结构。
- 中间层是否有可见脑区结构。

### 第 2 步：看 Stage1（预处理 + 血管先验）
- `mock/output/<case_id>/stage1/figs/report/stage1_raw_mid.png`
- `mock/output/<case_id>/stage1/figs/report/stage1_norm_mid.png`
- `mock/output/<case_id>/stage1/figs/report/stage1_vessel_prior_overlay.png`

你要看什么：
- 标准化后对比原图，是否更清晰、对比更稳定。
- 先验叠加图中彩色高响应区域是否主要落在血管样区域。

### 第 3 步：看 Stage2（候选生成 + 距离规则）
- `mock/output/<case_id>/stage2/figs/report/stage2_candidate_scatter.png`
- `mock/output/<case_id>/stage2/figs/report/stage2_candidate_per_class.png`
- `mock/output/<case_id>/stage2/figs/report/stage2_distance_filter_counts.png`
- `mock/output/<case_id>/stage2/figs/report/stage2_distance_hist_all.png`
- `mock/output/<case_id>/stage2/figs/report/stage2_roi_patch_grid.png`

你要看什么：
- 候选点是否集中在有效区域，而不是随机散落到背景。
- 各血管类别是否都有候选（避免类别完全缺失）。
- 距离过滤前后计数是否有变化（说明 Trick3 模拟起作用）。
- ROI 拼图是否可见结构、没有大量越界空白块。

### 第 4 步：看 Stage3（2.5D ROI 打分）
- `mock/output/<case_id>/stage3/figs/report/stage3_top_roi_scores.png`
- `mock/output/<case_id>/stage3/figs/report/stage3_top_evidence_grid.png`

你要看什么：
- Top ROI 分数是否有层次差异（不是全部一样）。
- Top 证据 patch 是否呈现一定形态多样性，而不是同一噪声模板。

### 第 5 步：看 Stage4（病例级 14 维结果）
- `mock/output/<case_id>/stage4/figs/report/stage4_pred14_bar.png`
- `mock/output/<case_id>/stage4/figs/report/stage4_max_vs_topk.png`

你要看什么：
- 14 维概率是否都在 `[0, 1]`。
- `Aneurysm Present` 是否等于 13 个位置概率的最大值。
- `max` 和 `top-k mean` 的差异是否合理（top-k 通常更平滑、更稳健）。

### 最后核对文件（数值结果）
- `mock/output/<case_id>/stage4/artifacts/case_pred.json`
- `mock/output/<case_id>/stage4/artifacts/submission_like.csv`

如果你只关心最终结果，这两个文件就是终点。
