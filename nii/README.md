# NII 解码与血管查看指南（小白版）

这份文档帮助你快速看懂 `nii/` 目录的解码输出：
- 原始 NIfTI 基础解码（切片图 + 元信息）
- 血管可视化输出（血管掩码 + 叠加图 + 坐标 JSON）

---

## 1. 你会用到的脚本

- `nii/decode_nii.py`
  - 作用：把 `.nii/.nii.gz` 解码成可读图片，并导出基础 `meta.json`。
- `nii/view_vessels_from_nii.py`
  - 作用：基于 HU 阈值生成粗血管掩码，导出血管叠加图与坐标 JSON。

---

## 2. 一键解码 NII（基础版）

```bash
python nii/decode_nii.py \
  --nii-path nii/your_case.nii \
  --output-dir nii/decoded \
  --num-grid-slices 9
```

输出目录示例：
`nii/decoded/<case_id>/`

主要文件：
- `axial_mid.png`：轴位中间层
- `coronal_mid.png`：冠状位中间层
- `sagittal_mid.png`：矢状位中间层
- `axial_grid.png`：轴位多层网格图
- `meta.json`：shape、spacing、affine、强度统计

---

## 3. 查看血管（增强版）

```bash
python nii/view_vessels_from_nii.py \
  --nii-path nii/your_case.nii \
  --output-dir nii/decoded \
  --vessel-low-hu 120 \
  --vessel-high-hu 700 \
  --num-grid-slices 9
```

输出目录示例：
`nii/decoded/<case_id>/vessel_view/`

主要文件：
- `vessel_mask.npy`：3D 二值血管掩码（0/1）
- `axial_vessel_overlay.png`：轴位血管叠加图
- `coronal_vessel_overlay.png`：冠状位血管叠加图
- `sagittal_vessel_overlay.png`：矢状位血管叠加图
- `axial_vessel_overlay_grid.png`：多层轴位叠加网格
- `axial_mip_vessel_overlay.png`：MIP 叠加图
- `vessel_summary.json`：阈值、体素占比、输出路径
- `vessel_coordinates.json`：血管坐标点（体素坐标 + 物理坐标）

---

## 4. 推荐看图顺序（最省时间）

1. 先看 `axial_grid.png`
- 确认原始体数据连续性，有无明显断层/空白。

2. 再看 `axial_vessel_overlay_grid.png`
- 看血管掩码是否主要在脑内亮血管区域。
- 如果大面积飘到脑外或骨结构，阈值可能不合适。

3. 看 `axial_mip_vessel_overlay.png`
- 快速判断整体血管树形态是否完整。
- 如果太稀疏：尝试降低 `--vessel-low-hu`（如 100）。
- 如果噪声太多：尝试提高 `--vessel-low-hu` 或降低 `--vessel-high-hu`。

4. 最后看三视图中间层
- `axial_vessel_overlay.png`
- `coronal_vessel_overlay.png`
- `sagittal_vessel_overlay.png`

用于检查不同方向是否都合理。

---

## 5. 怎么读 JSON

### 5.1 `meta.json`（基础解码）
关键字段：
- `shape_xyz`：体素维度
- `voxel_spacing`：体素间距（mm）
- `affine`：体素坐标 -> 物理坐标变换矩阵
- `intensity_stats`：强度分布（min/max/mean/std）

### 5.2 `vessel_summary.json`（血管统计）
关键字段：
- `threshold_hu.low/high`：当前血管阈值
- `vessel_voxels`：血管体素数量
- `vessel_ratio`：血管体素占总体素比例

### 5.3 `vessel_coordinates.json`（坐标详情）
关键字段：
- `num_points`：血管点总数
- `vessel_coordinates`：点列表
  - `voxel_xyz`：数组索引坐标 `(x,y,z)`
  - `world_xyz_mm`：物理坐标（mm）

---

## 6. 常见问题排查

### 问题1：血管区域太少
- 现象：`vessel_ratio` 很低，图上只剩少量碎点。
- 处理：降低 `--vessel-low-hu`，例如 `120 -> 100`。

### 问题2：血管区域太多（噪声）
- 现象：脑外或骨性结构被大量标红。
- 处理：提高 `--vessel-low-hu`（如 `150`）或降低 `--vessel-high-hu`（如 `600`）。

### 问题3：JSON 太大
- 原因：`vessel_coordinates.json` 包含所有血管点。
- 建议：后续可按需抽样或分块保存。

---

## 7. 当前案例快速入口

已处理案例：
`1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381`

可直接打开：
- `nii/decoded/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/meta.json`
- `nii/decoded/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/axial_grid.png`
- `nii/decoded/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/vessel_view/axial_vessel_overlay_grid.png`
- `nii/decoded/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381/vessel_view/vessel_coordinates.json`
