## TL;NR
```
export PYTHONPATH=/mnt/extra/Point-R1
```

## 项目简介

**Point-R1-data** 仓库主要用于：

- **离屏渲染 GLB 模型** 获取多视角 RGB 图像与深度图（基于 Open3D）。
- **从多视角图像提取 DINO 特征**，并**反投影到点云**，进行特征可视化与分析。
- **为 Point-R1 主项目做数据预处理**：将多视角 DINO 特征投影到 3D 点云上，得到 3D 特征分布模式，再将这些带有语义分布的投影结果（如彩色点云/渲染图）送入 LLM，用作 caption / instruction 生成的辅助信息。
- 提供与 PointLLM/Point-R1 训练相关的点云加载与预处理工具（`dataloader/` 与 `utils.py`）。

目录结构（只列出核心部分）：

- `example_material/`：示例数据与中间结果  
  - `npys/`：点云 `*.npy`（通常为 `N×3` 或 `N×6`）。  
  - `glbs/`：与点云对应的 `*.glb` 网格模型。  
  - `renders_o3d/`：由 `render_open3d_offscreen.py` 生成的多视角渲染结果。  
  - `dino_features/`：由 `extract_dino_features.py` 生成的特征与可视化结果。  
- `renderer_o3d.py`：Open3D 运行时渲染与相机几何工具。  
- `render_open3d_offscreen.py`：批量对 GLB 进行多视角渲染。  
- `extract_dino_features.py`：从多视角图像提取 DINO 特征并反投影到点云。  
- `dataloader/`：训练用数据加载与预处理脚本。  
- `checkpoints/`：本地 DINOv3 权重（需要自行准备或修改脚本中的路径）。  

## 环境依赖（简要）

- Python 3.8+  
- Open3D（推荐 `open3d>=0.17`）  
- PyTorch、torchvision  
- `transformers`（用于加载 DINOv3）  
- `scikit-learn`（PCA）、`numpy`、`Pillow`、`tqdm` 等常规依赖  

> 依赖安装可参考：
> 
> ```bash
> pip install open3d torch torchvision transformers scikit-learn pillow tqdm numpy
> ```

## 基本使用流程

### 1. 准备数据

- 将点云 `*.npy` 放在 `example_material/npys/`，文件名形如：  
  - `d1e62f1eb6334a1196b53d61529f472b_8192.npy`
- 将对应的 GLB 放在 `example_material/glbs/`，文件名形如：  
  - `d1e62f1eb6334a1196b53d61529f472b.glb`

### 2. 多视角渲染（可选，生成静态渲染）

```bash
cd /mnt/extra/Point-R1-data
python render_open3d_offscreen.py \
  --glb_dir example_material/glbs \
  --output_dir example_material/renders_o3d
```

运行后，每个 GLB 会在 `example_material/renders_o3d/<object_id>/` 下生成：

- `view_XXX.png`：RGB 渲染图  
- `view_XXX_depth.npy / view_XXX_depth.png`：对应深度图  

### 3. 提取 DINO 特征并反投影到点云

**运行时渲染模式（推荐调试时使用）：**

```bash
python extract_dino_features.py \
  --object_id d1e62f1eb6334a1196b53d61529f472b \
  --npy_dir example_material/npys \
  --output_dir example_material/dino_features \
  --device cuda \
  --use_runtime_rendering
```

**预渲染图像模式（使用第 2 步生成的渲染）：**

```bash
python extract_dino_features.py \
  --object_id d1e62f1eb6334a1196b53d61529f472b \
  --npy_dir example_material/npys \
  --render_dir example_material/renders_o3d \
  --output_dir example_material/dino_features \
  --device cuda \
  --no_runtime_rendering
```

运行结束后，主要输出包括：

- `example_material/dino_features/<object_id>_pca.ply`：点云上 DINO 特征的 PCA 伪彩色可视化。  
- `example_material/dino_features/<object_id>_features.npy`：每个点的 DINO 特征。  
- `example_material/dino_features/<object_id>_intermediate_2d/`：多视角 RGB、深度图、深度差异可视化等中间结果。  

## 基于 PartNeXt hierarchyList 的逐层命名与 Caption

脚本：`Point-R1-data/gt_tree_captioning.py`

它会：

- 读取 `hierarchyList` 并转为内部树结构（保留 `name/nodeId/maskId/children` 语义）
  - 注：在 `PartNeXt_data` 中，`hierarchyList` 实际上是一个 **字符串**（需要 `ast.literal_eval` 解析），且解析后通常为 **长度为 1 的 list**，其中唯一元素就是“真实树根”（`nodeId` 常为 `0`，`name` 是类别名，如 Chair/Glasses）。
- 对每个叶子 `maskId` 对应部件（PartNeXt part）从网格采样点，并在渲染图上生成高亮 mask
- 非叶子节点用子节点点集并集来高亮
- 对树做一次“**单子节点父名下沉**”修正：若某节点只有一个子节点，则把该节点名加入子节点候选名（`name_candidates`）
- 对每个节点让 MLLM **一次性输出最终部件名 + 非结构化 caption**（caption 含视觉/几何特征与功能推断）

示例：

```bash
python Point-R1-data/gt_tree_captioning.py \
  --partnext_glb_dir /path/to/PartNeXt/data/data \
  --partnext_ann_dir /path/to/PartNeXt/PartNeXt \
  --partnext_object_id b3e33144d8224385a2036b431e1b1451 \
  --output_dir outputs/partnext_tree_captions \
  --dry_run
```

## 点云与 GLB 加载及坐标系重定向

### GLB 加载与归一化

- GLB 通过 `renderer_o3d.Open3DRenderer.load_model(glb_path)` 加载为 `TriangleMeshModel`。  
- 使用 `Open3DRenderer.normalize_model(model)` 对整个模型进行统一归一化：  
  - 计算所有 mesh 的 **整体 Axis-Aligned Bounding Box (AABB)**。  
  - 将模型平移到以 AABB 中心为原点：  
    - \( \mathbf{p}' = \mathbf{p} - \mathbf{c}_{\text{bbox}} \)  
  - 以最大边长 `max_extent` 进行各向同性缩放到单位立方体：  
    - \( \mathbf{p}'' = \mathbf{p}' / \text{max\_extent} \)  
- 归一化后的模型大致位于 \([-0.5, 0.5]^3\) 附近，中心在原点附近，用于后续统一相机采样与渲染。

### 点云加载与坐标轴重定向

在 `extract_dino_features.py` 中：

- 点云通过 `load_pointcloud(npy_path)` 加载，得到原始坐标 \((x_p, y_p, z_p)\)。  
- 代码中明确给出 **原始点云坐标系与 GLB/mesh 坐标系的对应关系**：

> 原始点云坐标系说明：  
> - 原始点云的 **Y 轴正方向** = mesh 采样点的 **Z 轴负方向**  
> - 原始点云的 **Z 轴正方向** = mesh 的 **Y 轴正方向**

因此，从点云坐标 \((x_p, y_p, z_p)\) 映射到 mesh 坐标 \((x_m, y_m, z_m)\) 的关系为：

- \( x_m = x_p \)  
- \( y_m = z_p \)  
- \( z_m = -y_p \)  

即在代码中实现为：

- `points_aligned[:, 0] = points[:, 0]`  
- `points_aligned[:, 1] = points[:, 2]`  
- `points_aligned[:, 2] = -points[:, 1]`  

这一变换将点云对齐到与 GLB mesh 一致的右手坐标系中。

### 点云归一化与 GLB 的关系

点云在坐标轴重定向后，还会做一次**独立的归一化**（与 GLB 的归一化思路类似）：

- 根据对齐后的点云计算 AABB：  
  - 取 `min_bound`, `max_bound`，中心 \(\mathbf{c}_{\text{pc}}\) 与最大边长 `max_extent`。  
- 点云平移并缩放：

\[
\mathbf{p}_{\text{pc}}' = (\mathbf{p}_{\text{aligned}} - \mathbf{c}_{\text{pc}}) / \text{max\_extent}
\]

这意味着 GLB 与点云各自用自己的 AABB 做归一化（而不是共用一个变换矩阵），  
在数据来源一致的前提下，两者在完成轴对齐与归一化后，会自然落在同一尺度和相近中心下，  
从而便于使用统一的相机参数进行渲染和投影；若源数据本身存在偏移或缩放不一致，会在后文的 **fitness 评价** 中体现出来。

### 重定向后的 fitness 计算原理

在 `process_single_object` 中，为了检查“**轴对齐 + 归一化**”之后点云和 GLB 的一致性，脚本会：

1. 从归一化后的 GLB mesh 中均匀采样若干点：  
   - `mesh_sampled = mesh_for_eval.sample_points_uniformly(number_of_points=10000)`  
2. 使用归一化后的点云构造 `PointCloud`：  
   - `pcd_eval.points = Vector3dVector(points)`  
3. 调用 Open3D 的 `evaluate_registration` 进行**无配准变换的对齐质量评估**：

```python
eval_res = o3d.pipelines.registration.evaluate_registration(
    pcd_eval,
    mesh_sampled,
    max_corr_dist,   # 例如 0.05
    np.eye(4),       # 单位变换，假设两者已经在同一坐标系
)
```

- **`fitness`**：在给定阈值 `max_corr_dist` 内，源点云中有多少比例的点能找到对应的匹配点（0~1）。  
- **`inlier_rmse`**：所有内点匹配对的 RMSE 误差。  

解释：

- 这里**不进行 ICP 优化**，而是直接用单位矩阵评估，这意味着：  
  - 若 `fitness` 高且 `inlier_rmse` 小，说明“轴重定向 + 两边各自归一化”的策略使两者几何上已经高度对齐。  
  - 若 `fitness` 很低或 RMSE 很大，则表明点云与 GLB 在当前坐标系下存在明显几何偏差，需要检查原始数据或重定向逻辑。  
- 该结果只用于**质量检查与调试打印**，不会影响后续 DINO 特征提取/反投影流程。

## 相机内参与外参的计算原理

整个项目中，相机参数的核心逻辑集中在 `renderer_o3d.py` 与 `extract_dino_features.py` 中。

### 视角采样（相机中心的位置）

在 `renderer_o3d.sample_view_points(radius, partition)` 中：

- 以**球面采样**的方式在半径为 `radius` 的球面上采样相机位置：  
  - 使用 \(\phi \in [0, 2\pi)\)、\(\theta \in (0, \pi]\) 的网格生成点。  
  - 将球坐标转换为笛卡尔坐标：

\[
\begin{aligned}
x &= r \sin\theta \cos\phi \\
y &= r \cos\theta \\
z &= r \sin\theta \sin\phi
\end{aligned}
\]

- 额外加入极点 `[0, radius, 0]` 和 `[0, -radius, 0]`，并对所有点做一个轻微旋转，避免数值退化。  
- 这些点就是后续“相机位置” `viewpoint`（记为 \(\mathbf{e}\)）。

在 `render_open3d_offscreen.py` 中，会根据 GLB 的 AABB 调用 `calculate_optimal_radius`，用简单几何关系：

\[
\text{radius} = \frac{\text{模型最大尺寸}/2}{\tan(\text{fov}/2)} \times \text{margin\_factor}
\]

来保证模型完全进入视野。

### 相机内参（PinholeCameraIntrinsic）

在 `renderer_o3d.render_with_depth` 和 `extract_dino_features.save_camera_params` / `process_single_object` 中，  
内参与“**给定成像分辨率 + FOV 反推焦距**”的方式计算：

- 图像宽高：`width = height = image_size`。  
- 设垂直/水平视场角 `fov = 60°`，则

\[
f_x = f_y = \frac{\text{width}}{2 \tan(\text{fov}/2)}, \quad
c_x = \frac{\text{width}}{2}, \quad
c_y = \frac{\text{height}}{2}
\]

对应的内参矩阵：

\[
K =
\begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
\]

代码中由 `create_camera_intrinsic_from_params` 构造 `o3d.camera.PinholeCameraIntrinsic` 对象。  

### 相机外参（Look-at 视图矩阵）

外参矩阵由 `create_camera_extrinsic_from_viewpoint(viewpoint, center, up)` 生成，其逻辑等价于经典的 **look-at** 视图矩阵：

- 输入：
  - `eye`：相机位置 \(\mathbf{e}\)，来源于 `sample_view_points`。  
  - `at`：观察中心 \(\mathbf{c}\)，通常为模型/点云的中心（默认 `[0,0,0]` 或通过 AABB 估计）。  
  - `up`：世界上方向（默认 `[0,1,0]`）。  

计算步骤：

1. 前向向量（从相机指向目标）：

\[
\mathbf{f} = \frac{\mathbf{c} - \mathbf{e}}{\|\mathbf{c} - \mathbf{e}\|}
\]

2. 右向量：

\[
\mathbf{r} = \frac{\mathbf{f} \times \mathbf{up}}{\|\mathbf{f} \times \mathbf{up}\|}
\]

若 `up` 与 `f` 共线则重新设定 `up` 再计算。  

3. 重新正交化上向量：

\[
\mathbf{u} = \frac{\mathbf{f} \times \mathbf{r}}{\|\mathbf{f} \times \mathbf{r}\|}
\]

4. 构造旋转矩阵与平移向量（世界坐标 \(\to\) 相机坐标）：

\[
R_{\text{cam}} =
\begin{bmatrix}
\mathbf{r}^\top \\
\mathbf{u}^\top \\
\mathbf{f}^\top
\end{bmatrix}, \quad
\mathbf{t}_{\text{cam}} = - R_{\text{cam}} \, \mathbf{e}
\]

最终外参矩阵：

\[
T_{\text{world}\to\text{cam}} =
\begin{bmatrix}
R_{\text{cam}} & \mathbf{t}_{\text{cam}} \\
0 & 1
\end{bmatrix}
\]

代码中即：

- `R_cam = np.array([right, up_vec, forward])`  
- `t_cam = -R_cam @ eye`  
- 用这两者填充 4×4 的 `extrinsic`。

### 3D 点到像素坐标与深度的一致性

在 `renderer_o3d.project_points_to_image_with_depth` 中，使用上述内外参将 3D 点投影到像素坐标：

1. 齐次坐标变换到相机坐标系：

\[
\mathbf{p}_c = R_{\text{cam}} \mathbf{p}_w + \mathbf{t}_{\text{cam}}
\]

2. 透视投影到图像平面（忽略畸变）：

\[
\begin{aligned}
u &= f_x \frac{X_c}{Z_c} + c_x \\
v &= f_y \frac{Y_c}{Z_c} + c_y
\end{aligned}
\]

3. **深度定义**：  
   - 这里将深度 **定义为相机坐标系的 z 分量的绝对值**：  
\[
d = |Z_c|
\]
   - 保证与 Open3D 的 `render_to_depth_image(z_in_view_space=True)` 输出语义一致：  
     当 `z_in_view_space=True` 时，Open3D 深度图直接存储的是 view-space（相机坐标系）下的 z 值，而不是 [0,1] 归一化的深度。

4. 可见性与成像范围：

- 有效点需同时满足：
  - \(Z_c > 0\)（在相机前方）；  
  - \(0 \le u < W, \ 0 \le v < H\)（在图像范围内）。  
- `project_points_to_image_with_depth` 返回：
  - `pixel_coords`：像素坐标 \([u, v]\)。  
  - `depths`：上述定义的深度 \(d\)。  
  - `valid_mask`：标记点是否在视锥与图像范围内。

### 深度图与点云的一致性检查（可见性/对齐）

在 `extract_dino_features.save_2d_intermediate_results` 中，为每个视角计算：

- **GLB 渲染深度图**：来自 Open3D 的渲染结果。  
- **点云深度图**：将点云投影到该视角下的图像平面，并在每个像素取最近点的深度。  
- 二者做差，生成差异可视化图（蓝红色 `seismic` colormap 叠加在 GLB 深度可视化上）。

同时，在 `backproject_features_to_points` 中，`check_visible_points_with_depth` 会利用 GLB 渲染的深度图，对每个点的深度进行局部比较：

- 若点的深度与深度图中该像素的深度差 \(|d_{\text{pc}} - d_{\text{glb}}|\) 在一定相对阈值以内（例如 5–10%），  
  则认为该点在当前视角下**真实可见**，否则视为被遮挡或不一致。  

这样，反投影时只会使用几何上合理的可见点，提高 DINO 特征与 3D 几何的一致性。

## 实践中的注意事项总结

- **坐标系与归一化：**
  - 使用本仓库的脚本时，**务必保证点云与 GLB 语义上是一一对应的同一对象**。  
  - 如果你自定义点云生成方式，请确认其坐标轴含义与当前代码假设一致（`[x_p, y_p, z_p] -> [x_m, z_p, -y_p]`）。  
  - 若发现 `fitness` 很低，可视化 `example_material/dino_features/<id>_intermediate_2d` 中的深度差异/叠加图，检查是否存在平移/旋转错误。  

- **相机参数一致性：**
  - 不要单独修改渲染阶段与投影阶段的 FOV 或分辨率，否则 GLB 与点云的深度将无法对齐。  
  - 若需要更改图像尺寸或 FOV，应同时修改 `renderer_o3d.py` 与 `extract_dino_features.py` 中相关参数，并重新生成渲染与特征。  

- **深度图语义与 `z_in_view_space=True`：**
  - `renderer_o3d.Open3DRenderer.render_with_depth` 中使用  
    `depth_image = self.renderer.render_to_depth_image(z_in_view_space=True)`，此时 Open3D 返回的是**相机坐标系下的 z 分量**。  
  - `project_points_to_image_with_depth` 也使用 `|Z_c|` 作为深度定义，以保证与该深度图严格对齐；如果修改 `z_in_view_space` 或自行处理深度图，必须保证两处深度语义一致，否则可见性判断与几何对齐都会出错。

- **DINO 权重路径：**
  - `extract_dino_features.py` 中默认使用本地 checkpoint 路径  
    `dinov3_location = "/mnt/extra/my_task/checkpoint/dinov3-vit7b16-pretrain-lvd1689m"`；  
    若本地路径不同或希望从 HuggingFace Hub 加载（如官方 dinov3），需要相应修改该变量。  

- **性能与稳定性：**
  - 多 GPU 模式会为每张卡加载一个 DINO 模型实例，显存占用较高，建议根据实际硬件调整 `num_gpus_to_use`。  
  - 渲染与 DINO 前向均可能较慢，建议先用小数量对象与较小 `partition` 进行验证。


