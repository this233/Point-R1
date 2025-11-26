"""
点云和 3D 模型可视化 Gradio 应用

功能：
1. Objaverse 点云可视化（Plot 点云可视化）
2. GLB 文件可视化（Model3D 3D 模型可视化）
3. ModelNet40 数据集可视化（Plot 点云可视化）
4. Bad Case 分析 (基于 analyse.json)

使用方法：
    python visualize_pointcloud_gradio.py --port 7860
"""

import argparse
import os
import sys
import shutil
import tempfile
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import cv2
import json
import clustering_utils  # 新增导入
from renderer_o3d import (
    Open3DRenderer,
    check_visible_points_with_depth,
    project_points_to_image_with_depth,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint,
    sample_view_points
)
try:
    from dataloader import load_objaverse_point_cloud, ModelNet
    from dataloader.utils import pc_norm
except ImportError as e:
    print(f"警告: 无法导入 PointLLM 模块: {e}")
    print("请确保 PointLLM 已正确安装或路径正确")
    load_objaverse_point_cloud = None
    ModelNet = None

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False
    print("警告: trimesh 未安装，GLB 可视化功能将受限")

try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    rendering = None
    OPEN3D_AVAILABLE = False
    print("警告: open3d 未安装，某些高级可视化功能将不可用")


ANALYSIS_FILE = "/mnt/extra/PointLLM/pointllm_evaluation/analyse.json"

def points_to_plotly(points, colors=None, title="点云可视化", point_size=1.5):
    """
    将点云转换为 plotly figure
    
    参数:
        points: numpy array, shape (N, 3) 或 (N, 6)
        colors: numpy array, shape (N, 3) 或 None
        title: 图表标题
        point_size: 点的大小（半径）
    返回:
        plotly figure
    """
    # 提取坐标和颜色
    if points.shape[1] >= 6:
        xyz = points[:, :3]
        if colors is None:
            colors = points[:, 3:6]
            # 归一化颜色到 [0, 1]
            if colors.max() > 1.0:
                colors = colors / 255.0
    else:
        xyz = points[:, :3]
        if colors is None:
            colors = np.ones((xyz.shape[0], 3)) * 0.5  # 默认灰色
    
    # 确保颜色在 [0, 1] 范围内
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    # 转换为 RGB 字符串格式
    color_data = (colors * 255).astype(int)
    color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in color_data]
    
    # 创建 plotly figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_strings,
                    opacity=0.8,
                )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True, title='X'),
                yaxis=dict(visible=True, title='Y'),
                zaxis=dict(visible=True, title='Z'),
                aspectmode='data',
            ),
            title=title,
            paper_bgcolor='rgb(255,255,255)',
            height=600,
        ),
    )
    
    return fig


def load_objaverse_pc(data_path, object_id, pointnum=8192, use_color=True, point_size=1.5):
    """
    加载 Objaverse 点云并返回 plotly figure
    
    参数:
        data_path: 数据路径
        object_id: 对象 ID
        pointnum: 点数
        use_color: 是否使用颜色
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if load_objaverse_point_cloud is None:
        error_msg = "错误: 无法导入 load_objaverse_point_cloud，请确保 PointLLM 已正确安装"
        print(error_msg)
        return None, error_msg
    
    try:
        point_cloud = load_objaverse_point_cloud(data_path, object_id, pointnum=pointnum, use_color=use_color)
        print(f"成功加载点云: {object_id}, 点数: {point_cloud.shape[0]}")

        # ===== 坐标重定向 & 自身归一化（与 extract_dino_features.py 保持一致）=====
        # 原始点云坐标系说明（来自 extract_dino_features.py）：
        #   - 原始点云的 Y 轴正方向 = mesh 采样点的 Z 轴负方向
        #   - 原始点云的 Z 轴正方向 = mesh 的 Y 轴正方向
        #
        # 设原始点 (x_p, y_p, z_p)，对应到 mesh 坐标 (x_m, y_m, z_m) 为：
        #   x_m = x_p
        #   y_m = z_p
        #   z_m = -y_p
        # 即：
        #   [x_m, y_m, z_m] = [x_p, z_p, -y_p]
        #
        # 同时使用点云自身做归一化，使其与 GLB 归一化后坐标范围一致
        if point_cloud.shape[1] >= 3:
            points = point_cloud[:, :3]

            # 坐标轴对齐
            points_aligned = np.empty_like(points)
            points_aligned[:, 0] = points[:, 0]          # X 保持不变
            points_aligned[:, 1] = points[:, 2]          # Y <- 原始 Z
            points_aligned[:, 2] = -points[:, 1]         # Z <- - 原始 Y

            # 自身归一化（与 extract_dino_features.py 中一致）
            min_bound = points_aligned.min(axis=0)
            max_bound = points_aligned.max(axis=0)
            center = (min_bound + max_bound) / 2.0
            extent = max_bound - min_bound
            max_extent = np.max(extent)

            if max_extent < 1e-6:
                print("警告：Objaverse 点云范围过小，跳过归一化，仅做坐标轴对齐")
                points_norm = points_aligned.astype(np.float32)
            else:
                scale = 1.0 / max_extent
                points_norm = (points_aligned - center) * scale
                points_norm = points_norm.astype(np.float32)
                print("已对 Objaverse 点云进行了坐标轴对齐并归一化（与 extract_dino_features.py 一致）")

            # 将归一化后的坐标写回 point_cloud，保留颜色等其它信息
            point_cloud = point_cloud.copy()
            point_cloud[:, :3] = points_norm
        
        fig = points_to_plotly(point_cloud, title=f"Objaverse 点云: {object_id}", point_size=point_size)
        return fig, f"成功加载点云: {object_id}, 点数: {point_cloud.shape[0]}"
    except Exception as e:
        error_msg = f"错误: 无法加载点云数据: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_glb_model(glb_path):
    """
    加载 GLB 文件并返回文件路径用于 Model3D 可视化
    将文件复制到临时目录以符合 Gradio 的安全要求
    
    参数:
        glb_path: GLB 文件路径
    返回:
        临时目录中的 GLB 文件路径或 None（如果出错）
    """
    if not TRIMESH_AVAILABLE:
        error_msg = "错误: trimesh 未安装，无法加载 GLB 文件"
        print(error_msg)
        return None, error_msg
    
    try:
        # 验证 GLB 文件是否可以加载
        scene = trimesh.load(glb_path)
        
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
        elif isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        else:
            error_msg = "错误: 无法识别的 GLB 文件格式"
            print(error_msg)
            return None, error_msg
        
        print(f"成功加载 GLB 文件: {glb_path}")
        print(f"包含 {len(meshes)} 个网格")
        
        # 将文件复制到临时目录（Gradio 允许的路径）
        temp_dir = '/mnt/extra/tmp'
        # 确保临时目录存在
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = os.path.basename(glb_path)
        temp_file_path = os.path.join(temp_dir, filename)
        
        # 如果临时文件已存在，先删除
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        # 复制文件到临时目录
        shutil.copy2(glb_path, temp_file_path)
        print(f"文件已复制到临时目录: {temp_file_path}")
        
        info_msg = f"成功加载 GLB 文件，包含 {len(meshes)} 个网格"
        return temp_file_path, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载 GLB 文件: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_modelnet_pc(index=0, split='test', use_color=False, config_path=None, point_size=1.5):
    """
    加载 ModelNet40 点云并返回 plotly figure
    
    参数:
        index: 数据索引
        split: 'train' 或 'test'
        use_color: 是否使用颜色
        config_path: 配置文件路径
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if ModelNet is None:
        error_msg = "错误: 无法导入 ModelNet，请确保 PointLLM 已正确安装"
        print(error_msg)
        return None, error_msg
    
    try:
        # 使用默认配置文件路径
        if config_path is None or config_path == "":
            config_path = 'dataloader/modelnet_config/ModelNet40.yaml'
        if not os.path.exists(config_path):
            error_msg = f"警告: 配置文件不存在: {config_path}\n请手动指定 config_path 或确保 PointLLM 路径正确"
            print(error_msg)
            return None, error_msg
        
        dataset = ModelNet(config_path=config_path, split=split, subset_nums=-1, use_color=use_color)
        
        index = int(index)  # 确保索引是整数
        if index < 0 or index >= len(dataset):
            error_msg = f"错误: 索引 {index} 超出范围，数据集大小为 {len(dataset)}"
            print(error_msg)
            return None, error_msg
        
        data_dict = dataset[index]
        point_cloud = data_dict['point_clouds'].numpy()
        label = data_dict['labels']
        label_name = data_dict['label_names']
        
        print(f"成功加载 ModelNet40 数据:")
        print(f"  索引: {index}")
        print(f"  类别: {label_name} (ID: {label})")
        print(f"  点数: {point_cloud.shape[0]}")
        
        title = f"ModelNet40 - {label_name} (索引: {index})"
        fig = points_to_plotly(point_cloud, title=title, point_size=point_size)
        info_msg = f"成功加载 ModelNet40 数据:\n索引: {index}\n类别: {label_name} (ID: {label})\n点数: {point_cloud.shape[0]}"
        return fig, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载 ModelNet40 数据: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_pca_pointcloud(ply_path, point_size=1.5):
    """
    加载PCA点云文件（.ply格式）并返回 plotly figure
    
    参数:
        ply_path: .ply文件路径
        point_size: 点的大小（半径）
    返回:
        plotly figure 或 None（如果出错）
    """
    if not OPEN3D_AVAILABLE:
        error_msg = "错误: open3d 未安装，无法加载 .ply 文件"
        print(error_msg)
        return None, error_msg
    
    try:
        if not os.path.exists(ply_path):
            error_msg = f"错误: 文件不存在: {ply_path}"
            print(error_msg)
            return None, error_msg
        
        if not ply_path.lower().endswith('.ply'):
            error_msg = "错误: 文件必须是 .ply 格式"
            print(error_msg)
            return None, error_msg
        
        # 使用open3d加载点云
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(pcd.points) == 0:
            error_msg = "错误: 点云文件为空"
            print(error_msg)
            return None, error_msg
        
        # 提取点云坐标
        points = np.asarray(pcd.points)
        
        # 提取颜色（如果有）
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # open3d的颜色范围是[0, 1]，plotly也需要[0, 1]
        else:
            # 如果没有颜色，使用默认灰色
            colors = np.ones((points.shape[0], 3)) * 0.5
        
        print(f"成功加载PCA点云:")
        print(f"  文件: {ply_path}")
        print(f"  点数: {points.shape[0]}")
        print(f"  有颜色: {pcd.has_colors()}")
        
        filename = os.path.basename(ply_path)
        title = f"PCA点云可视化: {filename}"
        fig = points_to_plotly(points, colors=colors, title=title, point_size=point_size)
        info_msg = f"成功加载PCA点云:\n文件: {filename}\n点数: {points.shape[0]}\n有颜色: {pcd.has_colors()}"
        return fig, info_msg
        
    except Exception as e:
        error_msg = f"错误: 无法加载PCA点云: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


def load_and_cluster_points(pc_path, feat_path, alpha, beta1, beta2, beta3, beta4, point_size=1.5):
    """
    加载点云和特征，执行聚类，并返回初始视图（层级1）
    同时返回状态数据供后续切换层级使用
    """
    try:
        # 1. 加载数据
        if not os.path.exists(pc_path):
            return None, None, f"错误: 点云文件不存在: {pc_path}"
        if not os.path.exists(feat_path):
            return None, None, f"错误: 特征文件不存在: {feat_path}"
            
        # 加载点云 (.npy)
        points = np.load(pc_path)
        if points.shape[1] >= 3:
            points = points[:, :3]
        
        # ===== 坐标重定向 & 自身归一化（与 extract_dino_features.py 保持一致）=====
        # 确保点云与 GLB 模型的坐标系一致
        points_aligned = np.empty_like(points)
        points_aligned[:, 0] = points[:, 0]          # X 保持不变
        points_aligned[:, 1] = points[:, 2]          # Y <- 原始 Z
        points_aligned[:, 2] = -points[:, 1]         # Z <- - 原始 Y
        
        # 保存原始对齐点云（未归一化）用于后续与 GLB 对齐
        points_aligned_raw = points_aligned.copy()

        min_bound = points_aligned.min(axis=0)
        max_bound = points_aligned.max(axis=0)
        center = (min_bound + max_bound) / 2.0
        extent = max_bound - min_bound
        max_extent = np.max(extent)

        if max_extent < 1e-6:
            print("警告：点云范围过小，跳过归一化")
            points = points_aligned.astype(np.float32)
        else:
            scale = 1.0 / max_extent
            points = (points_aligned - center) * scale
            points = points.astype(np.float32)
            print("已对聚类点云进行了坐标轴对齐并归一化")

        # 加载特征 (.npy)
        features = np.load(feat_path)
        
        if points.shape[0] != features.shape[0]:
            return None, None, f"错误: 点云数量 ({points.shape[0]}) 与特征数量 ({features.shape[0]}) 不匹配"
            
        # UI 传递过来的是 "Alpha"，现在我们将其解释为 KNN 的 K 值
        k_neighbors = int(10 + alpha * 500) # alpha=0.02 -> k=20, alpha=0.05 -> k=35
        if k_neighbors < 5: k_neighbors = 5
        if k_neighbors > 100: k_neighbors = 100
        
        print(f"开始聚类计算... Points: {points.shape}, Features: {features.shape}")
        print(f"K-Neighbors: {k_neighbors} (from alpha={alpha}), Betas: {[beta1, beta2, beta3, beta4]}")
        
        # 2. 执行聚类
        betas = [beta1, beta2, beta3, beta4]
        # 注意：perform_hierarchical_clustering 现在接受 k_neighbors 而不是 alpha
        clustering_results = clustering_utils.perform_hierarchical_clustering(
            points, features, k_neighbors, betas
        )
        
        # 3. 准备状态数据
        state_data = {
            'points': points,
            'points_raw_aligned': points_aligned_raw,
            'clustering_results': clustering_results,
            'point_size': point_size
        }
        
        # 4. 默认显示层级 1 (index 0)
        fig, msg = update_cluster_view(state_data, "层级 1")
        
        full_msg = f"聚类计算完成。\n点数: {points.shape[0]}\n{msg}"
        return fig, state_data, full_msg
        
    except Exception as e:
        error_msg = f"错误: 聚类过程出错: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, error_msg


def update_cluster_view(state_data, level_name):
    """
    根据选择的层级更新视图
    """
    if state_data is None:
        return None, "请先加载数据并计算聚类"
        
    points = state_data['points']
    clustering_results = state_data['clustering_results']
    point_size = state_data.get('point_size', 1.5)
    
    # 解析层级索引
    level_map = {"层级 1": 0, "层级 2": 1, "层级 3": 2, "层级 4": 3}
    level_idx = level_map.get(level_name, 0)
    
    labels = clustering_results.get(level_idx)
    if labels is None:
        return None, f"未找到 {level_name} 的聚类结果"
        
    # 生成颜色
    colors = clustering_utils.generate_cluster_colors(labels)
    
    # 统计信息
    n_clusters = len(np.unique(labels))
    
    title = f"特征聚类可视化 - {level_name} (簇数量: {n_clusters})"
    fig = points_to_plotly(points, colors=colors, title=title, point_size=point_size)
    
    info_msg = f"当前显示: {level_name}\n簇数量: {n_clusters}"
    return fig, info_msg


def get_viewpoint_from_angles(azimuth, elevation, radius):
    """根据方位角、仰角和半径计算相机位置"""
    # 角度转弧度
    theta = np.radians(90 - elevation)  # Elevation 90 -> theta 0 (pole)
    phi = np.radians(azimuth)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    
    return np.array([x, y, z])


def get_points_mask(points_2d, image_size, k_size=31):
    """
    从 2D 点生成平滑掩码
    已针对近距离稀疏点云进行增强
    """
    H, W = image_size, image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 1. 增大绘图半径 (6 -> 15)
    for x, y in points_2d:
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(mask, (int(x), int(y)), radius=8, color=255, thickness=-1)
            
    # 2. 增大闭运算核 (9 -> 25)
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. 高斯模糊
    mask_blurred = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    
    return mask_smooth


def get_robust_mask_from_far_view(renderer, points, direction, center, current_dist, intrinsic, image_size, zoom_factor=3.0):
    """
    核心优化：利用“远距离视角”生成致密 Mask，然后放大适配当前视角。
    解决近距离下点云稀疏导致的 Mask 破碎/颗粒化问题。
    """
    # 1. 虚拟拉远相机
    far_dist = current_dist * zoom_factor
    eye_far = center + direction * far_dist
    extrinsic_far = create_camera_extrinsic_from_viewpoint(eye_far, center=center)
    
    # 2. 在远距离下投影 (此时点云密集，Mask 连通性好)
    # 注意：这里只做投影，不需要真实渲染图像，速度很快
    pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
        points, intrinsic, extrinsic_far, image_size=image_size
    )
    
    valid_pixels = pixel_coords[valid_mask]
    
    if len(valid_pixels) == 0:
        return np.zeros((image_size, image_size), dtype=np.uint8)
        
    # 3. 生成远距离 Mask (使用较小的基础半径即可，因为点很密)
    # 这里使用 radius=8, kernel=15 保证基础连通性
    mask_far = get_points_mask(valid_pixels, image_size, k_size=15) 
    
    # 4. 将 Mask 放大 zoom_factor 倍
    H, W = image_size, image_size
    center_x, center_y = W / 2.0, H / 2.0
    
    # 构建仿射变换矩阵：以图像中心为缩放中心
    # M = [ [scale, 0, (1-scale)*cx], [0, scale, (1-scale)*cy] ]
    M = np.array([
        [zoom_factor, 0, (1 - zoom_factor) * center_x],
        [0, zoom_factor, (1 - zoom_factor) * center_y]
    ], dtype=np.float32)
    
    # 使用最近邻或线性插值放大 Mask
    mask_near = cv2.warpAffine(mask_far, M, (W, H), flags=cv2.INTER_LINEAR)
    
    # 二值化清理边缘
    _, mask_near = cv2.threshold(mask_near, 127, 255, cv2.THRESH_BINARY)
    
    return mask_near


def optimize_distance_for_cluster(renderer, cluster_points, viewpoint_dir, center, intrinsic, image_size, target_occupancy=0.7, min_dist_threshold=0.8):
    """
    寻找最佳距离，满足两个条件：
    1. (硬约束) 完整性：必须包含簇中绝大部分点 (>99%)，不能溢出画面。
    2. (硬约束) 安全距离：相机不能进入物体内部 (dist > min_dist_threshold)。
    3. (软目标) 占比：在满足前两者的前提下，Cluster Mask 占比尽可能接近 target_occupancy。
    """
    # 0. 预处理：剔除极端的离群点，防止相机被拉得太远
    # 计算所有点到中心的距离
    dists_to_center = np.linalg.norm(cluster_points - center, axis=1)
    # 只考虑 99% 的点用于计算包围盒约束
    limit_dist = np.percentile(dists_to_center, 99.5)
    core_mask = dists_to_center <= limit_dist
    core_points = cluster_points[core_mask]
    
    total_core_points = len(core_points)
    if total_core_points == 0: return max(2.0, min_dist_threshold * 1.5)
    
    # 1. 估算搜索范围
    max_radius = limit_dist
    fov = 60.0 # 假设 FOV
    # 理论上能看到球体的最小安全距离
    min_view_dist = max_radius / np.sin(np.radians(fov / 2.0))
    
    # 搜索范围：必须大于 min_dist_threshold (防止穿模)
    # 同时也参考 min_view_dist (防止看不全)
    start_dist = max(min_dist_threshold, min_view_dist * 0.6) 
    end_dist = max(start_dist * 3.0, min_view_dist * 4.0, 3.5)
    
    test_dists = np.linspace(start_dist, end_dist, 30) # 增加采样密度
    
    best_dist = end_dist
    best_score = -float('inf')
    found_valid = False
    
    for dist in test_dists:
        eye = center + viewpoint_dir * dist
        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=center)
        
        # 投影所有核心点
        pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
            core_points, intrinsic, extrinsic, image_size=image_size
        )
        
        # --- 硬约束：完整性检查 ---
        # 计算在画面内（且在相机前方）的点的比例
        visible_count = np.sum(valid_mask)
        completeness = visible_count / total_core_points
        
        # 如果有超过 0.5% 的核心点出界，则认为该距离太近，无效
        if completeness < 0.995:
            # 如果之前还没找到任何有效距离，但这个距离虽然出界但还可以接受(>95%)，暂存为保底
            if not found_valid and completeness > 0.95:
                 score = -100.0 + completeness * 100 # 惩罚分
                 if score > best_score:
                     best_score = score
                     best_dist = dist
            continue
            
        found_valid = True
        
        # --- 软目标：占比优化 ---
        valid_pixels = pixel_coords[valid_mask]
        if len(valid_pixels) > 0:
            # 计算包围盒占比
            min_xy = np.min(valid_pixels, axis=0)
            max_xy = np.max(valid_pixels, axis=0)
            
            # 增加一点 Padding 惩罚，不仅要在画面内，最好不要贴边
            margin = image_size * 0.02
            if min_xy[0] < margin or min_xy[1] < margin or \
               max_xy[0] > image_size - margin or max_xy[1] > image_size - margin:
                padding_penalty = 0.2
            else:
                padding_penalty = 0.0
                
            w = max_xy[0] - min_xy[0]
            h = max_xy[1] - min_xy[1]
            occupancy = (w * h) / (image_size * image_size)
            
            # 评分：越接近 target 越好
            score = 1.0 - abs(occupancy - target_occupancy) - padding_penalty
            
            if score > best_score:
                best_score = score
                best_dist = dist
    
    return best_dist


def check_mask_connectivity(mask):
    """检查 Mask 连通性"""
    if mask is None: return False, 0.0
    
    H, W = mask.shape
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    if num_labels < 2: return False, 0.0
    
    # 计算最大连通块占比
    max_size = 0
    total_fg = 0
    for i in range(1, num_labels):
        size = np.count_nonzero(labels == i)
        total_fg += size
        if size > max_size:
            max_size = size
            
    if total_fg == 0: return False, 0.0
    
    connectivity_ratio = max_size / total_fg
    occupancy = total_fg / (H * W)
    
    # 判定为连通：最大块占 98% 以上
    return connectivity_ratio > 0.98, occupancy

def draw_cluster_contour(image, points_2d, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制点云簇的轮廓
    使用增强的 Metaball 技术，应对近距离下的点云稀疏问题
    """
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 1. 绘制初始点 (增大半径以应对近距离稀疏)
    # 半径从 6 -> 15
    for x, y in points_2d:
        cv2.circle(mask, (int(x), int(y)), radius=15, color=255, thickness=-1)
    
    # 2. 增强闭运算连接空隙
    # 核大小从 9x9 -> 25x25，大幅增强缝合能力
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. 高斯模糊 + 阈值化 -> 产生平滑边缘
    # 模糊核也相应增大
    mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
    _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    
    # 4. 查找并绘制轮廓
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    
    # 5. 多边形拟合平滑
    smooth_contours = []
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        smooth_contours.append(approx)
        
    cv2.drawContours(result, smooth_contours, -1, color, thickness)
    return result

def generate_ureca_images(image, child_mask, parent_mask=None, color=(0, 255, 255), thickness=2, blur_sigma=30):
    """
    生成 URECA 风格的两张图:
    1. Context Image: 全局模糊背景 + 父节点区域(如果有)清晰 + 子节点轮廓
       (如果无父节点，则背景清晰，只画轮廓)
    2. Local Image: 全局模糊背景 + 子节点区域清晰 + Crop
    
    参数:
        image: 原始图像 (RGB)
        child_mask: 子节点掩码 (0/255)
        parent_mask: 父节点掩码 (0/255), 如果为 None 则假设无父节点
        color: 轮廓颜色
        thickness: 轮廓粗细
        blur_sigma: 模糊程度
    """
    H, W = image.shape[:2]
    
    # 确保掩码是 uint8
    if child_mask.dtype != np.uint8:
        child_mask = child_mask.astype(np.uint8)
    if parent_mask is not None and parent_mask.dtype != np.uint8:
        parent_mask = parent_mask.astype(np.uint8)

    # 1. 准备基础模糊图像
    image_blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma)
    
    # --- 生成图 1: Context Image (父节点背景 + 子节点轮廓) ---
    # 规则: 如果没有父节点，则全图清晰；如果有父节点，则全图模糊但父节点清晰
    if parent_mask is None:
        img_context = image.copy()
    else:
        img_context = image_blurred.copy()
        # 将父节点区域还原为清晰图像
        mask_p_bool = parent_mask > 0
        img_context[mask_p_bool] = image[mask_p_bool]

    # 在 Context Image 上绘制子节点轮廓
    contours, _ = cv2.findContours(child_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 平滑轮廓
    smooth_contours = []
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        smooth_contours.append(approx)
        
    cv2.drawContours(img_context, smooth_contours, -1, color, thickness)
    
    # --- 生成图 2: Local Image (子节点清晰 + Crop) ---
    img_local = image_blurred.copy()
    # 让子节点区域变清晰
    mask_c_bool = child_mask > 0
    img_local[mask_c_bool] = image[mask_c_bool]
    
    # Crop
    # 找到 Bounding Box
    x, y, w, h = cv2.boundingRect(child_mask)
    
    # 增加 padding
    pad_ratio = 0.2
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    
    # 确保 crop 区域有效
    if x2 > x1 and y2 > y1:
        img_local_crop = img_local[y1:y2, x1:x2]
    else:
        img_local_crop = img_local # Fallback

    return img_context, img_local_crop

def render_cluster_contour(state_data, glb_path, cluster_id_input, level_name, azimuth, elevation, radius, use_ureca_style=False):
    """渲染指定视角并绘制聚类轮廓"""
    print(f"DEBUG: render_cluster_contour called with glb={glb_path}, id={cluster_id_input}, level={level_name}, ureca={use_ureca_style}")
    
    if state_data is None:
        print("DEBUG: state_data is None")
        return None, None, "请先计算聚类"
    if glb_path is None or not os.path.exists(glb_path):
        print(f"DEBUG: GLB path invalid: {glb_path}")
        return None, None, "GLB 文件路径无效"
        
    # 直接使用自身归一化的点云
    points = state_data['points']
    clustering_results = state_data['clustering_results']
    print(f"DEBUG: points shape: {points.shape}")
    
    # 解析层级和 Cluster ID
    level_map = {"层级 1": 0, "层级 2": 1, "层级 3": 2, "层级 4": 3}
    level_idx = level_map.get(level_name, 0)
    labels = clustering_results.get(level_idx)
    print(f"DEBUG: level_idx: {level_idx}")
    
    if labels is None:
        print("DEBUG: labels is None")
        return None, None, "无效的聚类层级"
    
    # 自动处理 Cluster ID
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"DEBUG: unique_labels count: {len(unique_labels)}")
    try:
        target_cluster_id = int(cluster_id_input)
    except:
        target_cluster_id = -1
        
    if target_cluster_id == -1:
        target_cluster_id = unique_labels[np.argmax(counts)]
        msg_suffix = f"(自动选择最大簇: {target_cluster_id})"
    elif target_cluster_id not in unique_labels:
        print(f"DEBUG: target_cluster_id {target_cluster_id} not in labels")
        return None, None, f"Cluster ID {target_cluster_id} 不存在"
    else:
        msg_suffix = f"(簇 ID: {target_cluster_id})"
    
    print(f"DEBUG: target_cluster_id: {target_cluster_id}")
        
    # 获取目标簇点 (Child)
    cluster_indices = np.where(labels == target_cluster_id)[0]
    cluster_points = points[cluster_indices]
    print(f"DEBUG: cluster_points shape: {cluster_points.shape}")
    
    # 获取父簇点 (Parent)
    parent_points = None
    parent_cluster_id = -1
    if level_idx > 0:
        # 查找上一层级的 labels
        current_search_level = level_idx - 1
        while current_search_level >= 0:
            parent_labels = clustering_results.get(current_search_level)
            if parent_labels is not None:
                # 假设子簇包含于父簇中，取第一个点的父 label
                first_point_idx = cluster_indices[0]
                parent_cluster_id = parent_labels[first_point_idx]
                
                parent_indices = np.where(parent_labels == parent_cluster_id)[0]
                
                # 如果父簇点数 > 子簇点数，说明找到了包含关系的父簇
                if len(parent_indices) > len(cluster_indices):
                    parent_points = points[parent_indices]
                    msg_suffix += f", Parent ID: {parent_cluster_id} (Level {current_search_level+1})"
                    print(f"DEBUG: Found valid Parent ID: {parent_cluster_id} at Level {current_search_level+1}, points: {parent_points.shape[0]}")
                    break
                else:
                    print(f"DEBUG: Parent {parent_cluster_id} at Level {current_search_level+1} size {len(parent_indices)} == Child, skipping...")
                    current_search_level -= 1
            else:
                print(f"DEBUG: No labels for Level {current_search_level+1}")
                break
        
        if parent_points is None:
             print("DEBUG: No larger parent cluster found after traversing up.")
    
    # 渲染
    image_size = 800 # 适中的分辨率
    print(f"DEBUG: Setting up renderer with size {image_size}")
    renderer = Open3DRenderer(width=image_size, height=image_size)
    renderer.setup()
    
    try:
        print(f"DEBUG: Loading model from {glb_path}")
        model = renderer.load_model(glb_path)
        # GLB 自身归一化
        model, _ = renderer.normalize_model(model)
        
        viewpoint = get_viewpoint_from_angles(azimuth, elevation, radius)
        print(f"DEBUG: viewpoint: {viewpoint}")
        
        # 渲染 RGB 和 深度
        img_array, depth_map = renderer.render_with_depth(model, viewpoint, return_depth=True)
        print(f"DEBUG: Rendered image shape: {img_array.shape}, depth_map shape: {depth_map.shape}")
        
        # 投影参数
        fov = 60.0
        camera_params = {
            'intrinsic': {
                'width': image_size, 'height': image_size,
                'fx': image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
                'fy': image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
                'cx': image_size / 2.0, 'cy': image_size / 2.0,
                'fov': fov
            }
        }
        intrinsic = create_camera_intrinsic_from_params(camera_params)
        extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint)
        
        # 辅助函数：投影并获取 mask
        def project_and_get_mask(pts_3d):
            pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
                pts_3d, intrinsic, extrinsic, image_size=image_size
            )
            # 可见性检查
            is_visible = check_visible_points_with_depth(
                pts_3d, pixel_coords, depths, depth_map,
                use_relative_threshold=True, relative_threshold_ratio=0.05
            )
            visible_indices = np.where(valid_mask & is_visible)[0]
            print(f"DEBUG: project_and_get_mask - total: {len(pts_3d)}, visible: {len(visible_indices)}")
            if len(visible_indices) > 0:
                visible_pixels = pixel_coords[visible_indices]
                return get_points_mask(visible_pixels, image_size), visible_pixels
            return None, None

        # 1. 获取子节点 Mask
        print("DEBUG: Getting child mask...")
        child_mask, child_pixels = project_and_get_mask(cluster_points)
        
        if child_mask is None:
            print("DEBUG: child_mask is None (no visible points)")
            return img_array, None, f"渲染完成 {msg_suffix} - 该视角下簇不可见"

        # 2. 获取父节点 Mask (如果有)
        parent_mask = None
        if parent_points is not None:
            print("DEBUG: Getting parent mask...")
            parent_mask, _ = project_and_get_mask(parent_points)
        else:
             print("DEBUG: No parent_points provided for mask generation")
            
        if use_ureca_style:
            print("DEBUG: Generating URECA style images...")
            # URECA 风格: 返回两张图
            img_context, img_local = generate_ureca_images(
                img_array, child_mask, parent_mask, 
                color=(0, 255, 255), thickness=3, blur_sigma=30
            )
            msg_suffix += " [URECA Style]"
            return img_context, img_local, f"渲染完成 {msg_suffix}"
            
        else:
            print("DEBUG: Generating normal contour image...")
            # 传统风格: 仅绘制轮廓
            result_img = draw_cluster_contour(img_array, child_pixels, color=(0, 255, 0), thickness=3)
            return result_img, None, f"渲染完成 {msg_suffix}"
        
    except Exception as e:
        print(f"DEBUG: Exception in render_cluster_contour: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"渲染错误: {e}"
    finally:
        renderer.cleanup()

def auto_find_best_views(state_data, glb_path, cluster_id_input, level_name):
    # Placeholder for old function if referenced, but we don't use it in the main workflow anymore.
    # Keeping it simple or just return None to avoid errors if called.
    # Actually I will restore it fully just in case.
    pass


# ===================== New SoM Visual Prompt Functions =====================

def find_children_group(clustering_results, parent_level_idx, parent_id):
    """
    Given a parent cluster at a certain level, find its children at the next splitting level.
    Returns: (child_level_idx, child_ids, info_msg)
    """
    parent_labels = clustering_results.get(parent_level_idx)
    if parent_labels is None:
        return None, None, "Parent level not found"
        
    # Indices of points belonging to parent
    parent_indices = np.where(parent_labels == parent_id)[0]
    
    if len(parent_indices) == 0:
        return None, None, f"Parent ID {parent_id} is empty"
        
    # Traverse down levels to find split
    max_level = 3 # 0, 1, 2, 3
    
    print(f"DEBUG: Searching children for Parent {parent_id} (Level {parent_level_idx+1})...")
    
    for lvl in range(parent_level_idx + 1, max_level + 1):
        child_labels = clustering_results.get(lvl)
        if child_labels is None: 
            print(f"DEBUG: Level {lvl+1} data missing.")
            break
        
        # Get labels for the parent's points at this level
        current_labels = child_labels[parent_indices]
        unique_children = np.unique(current_labels)
        
        print(f"DEBUG:  - Checking Level {lvl+1}: Found {len(unique_children)} children ids: {unique_children}")

        # If we found more than 1 child (split happened), return immediately
        if len(unique_children) > 1:
            return lvl, unique_children, f"Found split at Level {lvl+1} ({len(unique_children)} children)"
            
        # If we are at the last level, we have to return whatever we have
        if lvl == max_level:
            msg = f"Reached bottom Level {lvl+1} with no split (Single child structure)."
            print(f"DEBUG: {msg}")
            return lvl, unique_children, msg
            
    return None, None, "No split found in deeper levels"

def get_optimal_text_location(mask):
    """
    使用距离变换寻找放置标签的最佳位置 (参考 SoM 实现)
    """
    # 确保 mask 是 uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Pad mask to handle edge cases
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    
    # Distance Transform: 计算每个前景像素到最近背景像素的距离
    # 这能找到 mask 内部"最深"的点，比质心更适合放置标签（特别是对于凹多边形）
    mask_dt = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 0)
    mask_dt = mask_dt[1:-1, 1:-1] # Remove padding
    
    max_dist = np.max(mask_dt)
    
    # 如果 mask 太小或全空
    if max_dist <= 0:
        # Fallback to centroid or image center
        M = cv2.moments(mask)
        if M["m00"] != 0:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            # Find any point
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) > 0:
                return int(np.mean(coords[:, 1])), int(np.mean(coords[:, 0]))
            return mask.shape[1]//2, mask.shape[0]//2
    
    # 找到最大距离的所有点
    coords_y, coords_x = np.where(mask_dt == max_dist)
    
    # 选择中心点
    cX = coords_x[len(coords_x)//2]
    cY = coords_y[len(coords_y)//2]
    
    return cX, cY

def draw_som_annotations(image, masks, labels, colors=None, occluded_masks=None):
    """
    Draw SoM style annotations:
    - Dim background (where no mask is present)
    - Draw translucent masks (VISIBLE parts)
    - Draw contours (VISIBLE parts)
    - Draw hatched/lighter masks (OCCLUDED parts, optional)
    - Draw numeric labels at centroids of VISIBLE parts
    """
    H, W = image.shape[:2]
    
    # 1. Combine masks to define foreground (only visible parts determine "focus")
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    for mask in masks:
        if mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
    # 2. Dim background
    result_img = image.copy().astype(np.float32)
    fg_bool = combined_mask > 0
    
    # Dim the background (non-mask area)
    result_img[~fg_bool] *= 0.3
    
    # 3. Draw each mask overlay
    overlay = result_img.copy()
    
    np.random.seed(42) # Consistent colors
    
    # Draw Occluded Masks First (Bottom Layer, subtler)
    if occluded_masks is not None:
        for i, (mask, label_id) in enumerate(zip(occluded_masks, labels)):
            if mask is None or np.sum(mask) == 0: continue
            
            # Use SAME color as visible part if provided
            if colors is None:
                # This case shouldn't happen if called from our main logic, but fallback just in case
                np.random.seed(label_id * 100) 
                color = np.random.randint(0, 255, 3).tolist() 
            else:
                color = colors[i]
                
            mask_bool = mask > 0
            
            # Style for Occluded:
            # Same color tone, but much lighter/transparent
            # Alpha 0.15 for color, blending with background
            overlay[mask_bool] = overlay[mask_bool] * 0.85 + np.array(color) * 0.15
            
    # Draw Visible Masks (Top Layer, clear)
    for i, (mask, label_id) in enumerate(zip(masks, labels)):
        if mask is None or np.sum(mask) == 0: continue
        
        # Generate color (if not provided)
        if colors is None:
            np.random.seed(label_id * 100)
            color = np.random.randint(0, 255, 3).tolist() # BGR
        else:
            color = colors[i]
            
        # Fill mask
        mask_bool = mask > 0
        overlay[mask_bool] = overlay[mask_bool] * 0.4 + np.array(color) * 0.6
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Smooth contours
        smooth_contours = []
        for cnt in contours:
             epsilon = 0.005 * cv2.arcLength(cnt, True)
             approx = cv2.approxPolyDP(cnt, epsilon, True)
             smooth_contours.append(approx)
        
        cv2.drawContours(overlay, smooth_contours, -1, color, 2) # Thicker contour
        
        # Draw Label
        cX, cY = get_optimal_text_location(mask)
        
        # Draw label with background box for visibility
        text = str(label_id)
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Box
        cv2.rectangle(overlay, (cX - text_w//2 - 4, cY - text_h//2 - 4), 
                      (cX + text_w//2 + 4, cY + text_h//2 + 4), (0,0,0), -1)
        # Text
        cv2.putText(overlay, text, (cX - text_w//2, cY + text_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
    result_img = np.clip(overlay, 0, 255).astype(np.uint8)
    return result_img

def draw_labels_only(image, masks, labels):
    """
    Draw only numeric labels at centroids (Left Image style)
    """
    result_img = image.copy()
    
    for mask, label_id in zip(masks, labels):
        if mask is None or np.sum(mask) == 0: continue
        
        cX, cY = get_optimal_text_location(mask)
            
        text = str(label_id)
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Box (Black)
        cv2.rectangle(result_img, (cX - text_w//2 - 4, cY - text_h//2 - 4), 
                      (cX + text_w//2 + 4, cY + text_h//2 + 4), (0,0,0), -1)
        # Text (White)
        cv2.putText(result_img, text, (cX - text_w//2, cY + text_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        
    return result_img

def render_sibling_group_views(state_data, glb_path, parent_level_name, parent_id):
    """
    Automatically find 4 views for the sibling group of the selected parent,
    and generate SoM style visual prompts.
    """
    if state_data is None: return [None]*12 + ["Please calculate clustering first"]
    
    level_map = {"层级 1": 0, "层级 2": 1, "层级 3": 2, "层级 4": 3}
    parent_level_idx = level_map.get(parent_level_name, 0)
    
    points = state_data['points']
    clustering_results = state_data['clustering_results']
    
    # 1. Find siblings
    child_level_idx, child_ids, msg = find_children_group(clustering_results, parent_level_idx, int(parent_id))
    
    if child_ids is None:
        return [None]*12 + [f"Error finding children: {msg}"]
        
    print(f"Found {len(child_ids)} children at Level {child_level_idx+1}: {child_ids}")
    
    child_labels = clustering_results[child_level_idx]
    
    # Collect points for all children (the "Target Group")
    # We want to optimize views for the UNION of these children
    group_indices = []
    group_point_child_ids = []
    for cid in child_ids:
        indices = np.where(child_labels == cid)[0]
        group_indices.extend(indices)
        group_point_child_ids.extend([cid] * len(indices))
    
    group_points = points[group_indices]
    group_point_child_ids = np.array(group_point_child_ids)
    group_center = np.mean(group_points, axis=0)
    
    # Color map for children
    # Pre-generate random colors for each child ID
    np.random.seed(123)
    color_map = {cid: np.random.randint(0, 255, 3).tolist() for cid in child_ids}
    
    # 2. Search for 4 best views
    image_size = 256
    renderer = Open3DRenderer(width=image_size, height=image_size)
    renderer.setup()
    
    try:
        model = renderer.load_model(glb_path)
        model, transform = renderer.normalize_model(model) # Get transform to know scale if needed
        renderer.upload_model(model)
        
        # Calculate safe distance threshold from model bbox
        # Model is normalized to unit cube (max extent = 1.0) centered at roughly (0,0,0)
        # Max radius from center is roughly sqrt(0.5^2 + 0.5^2 + 0.5^2) = 0.866
        # We set a safe threshold slightly larger than this
        model_safe_radius = 0.5
        
        fov = 60.0
        cam_params = {
            'intrinsic': {
                'width': image_size, 'height': image_size,
                'fx': image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
                'fy': image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
                'cx': image_size / 2.0, 'cy': image_size / 2.0,
                'fov': fov
            }
        }
        intrinsic = create_camera_intrinsic_from_params(cam_params)
        
        # Sample views
        view_points = sample_view_points(radius=1.0, partition=3) # More dense sampling
        candidates = []
        
        for vp in view_points:
            direction = vp / np.linalg.norm(vp)
            
            # Optimize distance for the WHOLE GROUP
            # Target occupancy 0.6 to allow some context
            # Pass model_safe_radius to prevent camera from going inside the object
            dist = optimize_distance_for_cluster(
                renderer, group_points, direction, group_center, intrinsic, image_size, 
                target_occupancy=0.6, min_dist_threshold=model_safe_radius
            )
            
            eye = group_center + direction * dist
            extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=group_center)
            
            # 1. Render depth map for occlusion checking
            # We must perform occlusion check, otherwise "visible" points might be behind the mesh
            _, depth_map = renderer.render_view(eye, center=group_center, return_depth=True)
            
            # 2. Project points to get coordinates and depths
            pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
                group_points, intrinsic, extrinsic, image_size=image_size
            )
            
            # 3. Check occlusion using depth map
            is_visible_depth = np.zeros(len(group_points), dtype=bool)
            fov_indices = np.where(valid_mask_fov)[0]
            
            if len(fov_indices) > 0:
                visible_mask_sub = check_visible_points_with_depth(
                    group_points[fov_indices], 
                    pixel_coords[fov_indices], 
                    depths[fov_indices], 
                    depth_map,
                    use_relative_threshold=True, 
                    relative_threshold_ratio=0.02 # Slightly stricter threshold
                )
                is_visible_depth[fov_indices] = visible_mask_sub
            
            valid_mask = is_visible_depth
            
            # 1. Basic Visibility Score
            visible_count = np.sum(valid_mask)
            total_count = len(group_points)
            visibility_ratio = visible_count / total_count
            
            if visibility_ratio < 0.1: continue # Too few points visible
            
            # 2. Analyze Child Distribution (Separation & Overlap)
            # Re-calculate using FULL points (ignoring occlusion) for Overlap check
            # User Requirement: Overlap should be calculated based on the FULL mask, not just the visible part.
            # This prevents cases where a child is "visible" only because its front face is seen, but it's actually blocking another child behind it.
            
            full_pixels = pixel_coords[valid_mask_fov] # Points inside FOV
            full_cids = group_point_child_ids[valid_mask_fov]
            
            child_bboxes_full = {}
            child_centers_full = {}
            
            for cid in child_ids:
                mask_c = (full_cids == cid)
                pts_c = full_pixels[mask_c]
                
                if len(pts_c) < 5: continue
                
                min_xy = np.min(pts_c, axis=0)
                max_xy = np.max(pts_c, axis=0)
                center = np.mean(pts_c, axis=0)
                
                child_bboxes_full[cid] = (min_xy, max_xy)
                child_centers_full[cid] = center
                
            # Calculate Metrics using FULL bboxes
            total_overlap_ratio = 0.0
            total_separation = 0.0
            pair_count = 0
            
            cids_present = list(child_bboxes_full.keys())
            
            # Still calculate visibility ratio for score (using the previously computed occlusion-aware mask)
            child_visibility_ratio = len(cids_present) / len(child_ids) # Note: This is "in FOV" ratio now, but we want "visible" ratio
            
            # Re-calculate strict visibility ratio for scoring
            visible_indices = np.where(valid_mask)[0]
            vis_cids = group_point_child_ids[visible_indices]
            unique_vis_cids = np.unique(vis_cids)
            strict_child_vis_ratio = len(unique_vis_cids) / len(child_ids)

            if len(cids_present) < 2:
                avg_overlap = 0.0
                norm_separation = 0.0
            else:
                for i in range(len(cids_present)):
                    for j in range(i+1, len(cids_present)):
                        cid1 = cids_present[i]
                        cid2 = cids_present[j]
                        
                        b1_min, b1_max = child_bboxes_full[cid1]
                        b2_min, b2_max = child_bboxes_full[cid2]
                        
                        # Intersection
                        ix_min = max(b1_min[0], b2_min[0])
                        iy_min = max(b1_min[1], b2_min[1])
                        ix_max = min(b1_max[0], b2_max[0])
                        iy_max = min(b1_max[1], b2_max[1])
                        
                        iw = max(0, ix_max - ix_min)
                        ih = max(0, iy_max - iy_min)
                        intersection = iw * ih
                        
                        # Union
                        area1 = (b1_max[0] - b1_min[0]) * (b1_max[1] - b1_min[1])
                        area2 = (b2_max[0] - b2_min[0]) * (b2_max[1] - b2_min[1])
                        union = area1 + area2 - intersection
                        
                        if union > 0:
                            total_overlap_ratio += intersection / union
                        
                        # Separation
                        dist_sep = np.linalg.norm(child_centers_full[cid1] - child_centers_full[cid2])
                        total_separation += dist_sep
                        pair_count += 1
                        
                avg_overlap = total_overlap_ratio / pair_count if pair_count > 0 else 0
                norm_separation = (total_separation / pair_count) / image_size if pair_count > 0 else 0

            # Final Score Formula
            # Priority 1: Avoid Overlap (Penalty)
            # Priority 2: Ensure Visibility (Occlusion Check enabled)
            # Priority 3: Maximize Separation
            
            # Tuned weights based on user feedback "Maximize visibility" & "High Quality"
            # Heavily penalize overlap to enforce strict separation of full masks
            score = (visibility_ratio * 8.0) + \
                    (strict_child_vis_ratio * 5.0) + \
                    (norm_separation * 4.0) - \
                    (avg_overlap * 30.0)
                
            candidates.append({
                'score': score,
                'direction': direction,
                'dist': dist,
                'eye': eye,
                'stats': {
                    'vis': visibility_ratio,
                    'child_vis': strict_child_vis_ratio,
                    'sep': norm_separation,
                    'overlap': avg_overlap
                }
            })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Debug: Print top candidates
        print("DEBUG: Top 5 Candidates:")
        for i, cand in enumerate(candidates[:5]):
             stats = cand.get('stats', {})
             print(f"  #{i+1}: Score={cand['score']:.2f} | Vis={stats.get('vis',0):.2f}, ChildVis={stats.get('child_vis',0):.2f}, Sep={stats.get('sep',0):.2f}, Overlap={stats.get('overlap',0):.2f}")

        # --- Quality Check & Filtering ---
        # Thresholds (Stricter based on user request for high quality)
        # Overlap threshold tightened to ensure full masks are clearly distinguished
        MAX_ALLOWED_OVERLAP = 0.2   # Reject if overlap > 25%
        MIN_ALLOWED_VISIBILITY = 0.18 # Reject if vis < 20%
        MIN_CHILD_VISIBILITY = 0.75   # Reject if < 60% of children are visible
        
        valid_candidates = []
        for cand in candidates:
            stats = cand.get('stats', {})
            if stats.get('overlap', 1.0) <= MAX_ALLOWED_OVERLAP and \
               stats.get('vis', 0.0) >= MIN_ALLOWED_VISIBILITY and \
               stats.get('child_vis', 0.0) >= MIN_CHILD_VISIBILITY:
                valid_candidates.append(cand)
                
        if len(valid_candidates) == 0:
            best_cand = candidates[0] if candidates else None
            best_overlap = best_cand.get('stats', {}).get('overlap', 0) if best_cand else 0
            msg = f"视图生成失败：无法找到满足高质量要求的视角。\n(Best Overlap: {best_overlap:.1%}, Threshold: {MAX_ALLOWED_OVERLAP:.1%})"
            print(f"DEBUG: Rejected all candidates. Best overlap: {best_overlap}")
            return [None]*12 + [msg]

        # Select 6 distinct views from VALID candidates
        final_views = []
        for cand in valid_candidates:
            if len(final_views) >= 6: break
            
            is_distinct = True
            for selected in final_views:
                if np.dot(cand['direction'], selected['direction']) > 0.8: # Threshold for distinctness (Was 0.7=45deg, Now 0.9=25deg)
                    is_distinct = False
                    break
            if is_distinct:
                final_views.append(cand)
        
        # If we found fewer than 6 valid distinct views, we just return what we have.
        # We do NOT fill with bad views.
        if len(final_views) == 0:
             return [None]*12 + ["筛选后未找到独立视角"]
             
        print(f"DEBUG: Selected {len(final_views)} valid views.")

        # 3. Render High-Res Final Images
        renderer.cleanup()
        renderer = None
        
        final_size = 800
        renderer_final = Open3DRenderer(width=final_size, height=final_size)
        renderer_final.setup()
        renderer_final.upload_model(model) # Re-upload
        
        cam_params['intrinsic']['width'] = final_size
        cam_params['intrinsic']['height'] = final_size
        cam_params['intrinsic']['fx'] = final_size / (2.0 * np.tan(np.radians(fov) / 2.0))
        cam_params['intrinsic']['fy'] = final_size / (2.0 * np.tan(np.radians(fov) / 2.0))
        cam_params['intrinsic']['cx'] = final_size / 2.0
        cam_params['intrinsic']['cy'] = final_size / 2.0
        intrinsic_final = create_camera_intrinsic_from_params(cam_params)
            
        output_pairs = [] # Will contain (Left, Right) for each view
            
        for view in final_views:
            # Render base image
            img_base, _ = renderer_final.render_view(view['eye'], center=group_center)
            
            # Generate masks for EACH child
            child_labels_list = []
            child_colors = []
            
            # 1. Get strict visible mask for current view
            # Note: Using create_camera_extrinsic_from_viewpoint is correct here as it matches render_view logic
            pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
                group_points, intrinsic_final, create_camera_extrinsic_from_viewpoint(view['eye'], center=group_center), image_size=final_size
            )
            # Re-render depth for final size to check occlusion
            _, depth_map_final = renderer_final.render_view(view['eye'], center=group_center, return_depth=True)
            
            is_visible_strict = np.zeros(len(group_points), dtype=bool)
            fov_indices = np.where(valid_mask_fov)[0]
            if len(fov_indices) > 0:
                 vis_sub = check_visible_points_with_depth(
                    group_points[fov_indices], 
                    pixel_coords[fov_indices], 
                    depths[fov_indices], 
                    depth_map_final,
                    use_relative_threshold=True, 
                    relative_threshold_ratio=0.02
                 )
                 is_visible_strict[fov_indices] = vis_sub
                 
            # Now for each child, generate mask based on VISIBLE points
            child_masks_visible = []
            child_masks_occluded = []
            
            for cid in child_ids:
                # Record metadata for drawing
                child_labels_list.append(cid)
                child_colors.append(color_map[cid])

                # Find indices relative to the FULL group cloud
                c_local_mask = (group_point_child_ids == cid)
                c_points_all = group_points[c_local_mask]
                
                # Get visible subset for this child
                c_vis_mask = is_visible_strict[c_local_mask]
                c_points_vis = c_points_all[c_vis_mask]
                c_points_occ = c_points_all[~c_vis_mask]
                
                # 1. Generate Visible Mask
                if len(c_points_vis) > 0:
                    mask_vis = get_robust_mask_from_far_view(
                        renderer_final, c_points_vis, view['direction'], group_center, 
                        view['dist'], intrinsic_final, final_size, zoom_factor=2.0
                    )
                else:
                    mask_vis = np.zeros((final_size, final_size), dtype=np.uint8)
                child_masks_visible.append(mask_vis)
                
                # 2. Generate Occluded Mask (if requested)
                if len(c_points_occ) > 0:
                     # Use slightly larger kernel or different processing if needed, 
                     # but standard robust mask is fine.
                     mask_occ = get_robust_mask_from_far_view(
                        renderer_final, c_points_occ, view['direction'], group_center, 
                        view['dist'], intrinsic_final, final_size, zoom_factor=2.0
                    )
                else:
                    mask_occ = np.zeros((final_size, final_size), dtype=np.uint8)
                child_masks_occluded.append(mask_occ)
            
            # Generate Right Image (SoM)
            img_right = draw_som_annotations(
                img_base, 
                child_masks_visible, 
                child_labels_list, 
                child_colors,
                occluded_masks=child_masks_occluded # Pass occluded masks
            )
            
            # Generate Left Image (Clean + Labels)
            img_left = draw_labels_only(img_base, child_masks_visible, child_labels_list)
            
            output_pairs.append(img_left)
            output_pairs.append(img_right)
            
        # Format camera info for LLM
        camera_info_str = f"Successfully generated {len(final_views)} views for parent {parent_id} (Level {parent_level_idx+1} -> {child_level_idx+1})\n\n"
        camera_info_str += "### Camera Viewpoints (Intuitive)\n"
        
        for i, view in enumerate(final_views):
            eye = view['eye']
            at = group_center
            
            # Calculate intuitive angles
            vec = eye - at
            dx, dy, dz = vec
            dist = np.linalg.norm(vec)
            
            # Elevation (Angle with XZ plane)
            # sin(elev) = y / dist
            elevation = np.degrees(np.arcsin(dy / dist)) if dist > 1e-6 else 0.0
            
            # Azimuth (Angle on XZ plane, from X axis)
            # 0: +X, 90: +Z, 180: -X, 270: -Z
            azimuth = np.degrees(np.arctan2(dz, dx))
            if azimuth < 0: azimuth += 360.0
            
            # Simple Direction Description (Simplified to Front/Back relative)
            # 0 (+X) -> Right; 90 (+Z) -> Front; 180 (-X) -> Left; 270 (-Z) -> Back
            # We want to categorize mainly into Front vs Back perspectives
            
            # Normalize azimuth to [0, 360)
            az = azimuth % 360
            
            if 45 <= az < 135:
                azimuth_desc = "Front"
            elif 135 <= az < 225:
                azimuth_desc = "Side"
            elif 225 <= az < 315:
                azimuth_desc = "Back"
            else: # 315-360 or 0-45
                azimuth_desc = "Side"
                
            # Add diagonal nuance if needed, or stick to user request "Back / Front and diagonals"
            # User asked for: "back / front 以及斜向前/后" (Back/Front and Diagonal Front/Back)
            # Let's refine:
            
            if 67.5 <= az < 112.5:
                azimuth_desc = "Front (+Z)"
            elif 112.5 <= az < 157.5:
                azimuth_desc = "Front Diagonal"
            elif 157.5 <= az < 202.5:
                azimuth_desc = "Side " # Pure side is less likely to be "Front/Back"
            elif 202.5 <= az < 247.5:
                azimuth_desc = "Back Diagonal"
            elif 247.5 <= az < 292.5:
                azimuth_desc = "Back (-Z)"
            elif 292.5 <= az < 337.5:
                azimuth_desc = "Back Diagonal"
            elif 337.5 <= az or az < 22.5:
                azimuth_desc = "Side"
            else: # 22.5 <= az < 67.5
                azimuth_desc = "Front Diagonal"
            
            # Map elevation to qualitative description
            # > 60: Top-down; 30~60: High Angle; -15~30: Eye-level; < -15: Low Angle
            if elevation > 60:
                elev_desc = "Top-down View"
            elif elevation > 30:
                elev_desc = "High Angle (Looking Down)"
            elif elevation > -15:
                elev_desc = "Eye-level View"
            else:
                elev_desc = "Low Angle (Looking Up)"
            
            # Combine into full description
            full_desc = f"{azimuth_desc}, {elev_desc}"

            camera_info_str += f"**View {i+1}:**\n"
            camera_info_str += f"- **Angle:** Azimuth {azimuth:.1f}°, Elevation {elevation:.1f}°\n"
            camera_info_str += f"- **Distance:** {dist:.3f}\n"
            camera_info_str += f"- **Description:** {full_desc}\n"
            camera_info_str += f"- **Raw Eye:** [{eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f}]\n"
            camera_info_str += f"- **Raw At:** [{at[0]:.3f}, {at[1]:.3f}, {at[2]:.3f}]\n\n"

        # Ensure we output 12 images (fill with None if fewer)
        while len(output_pairs) < 12:
            output_pairs.append(None)

        renderer_final.cleanup()
        renderer_final = None
        return [*output_pairs, camera_info_str]

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [None]*12 + [f"Error in rendering views: {e}"]
    finally:
        if renderer: renderer.cleanup()
        if 'renderer_final' in locals() and renderer_final: renderer_final.cleanup()


# ===================== Bad Case Analysis Functions =====================

def load_bad_cases():
    """Load bad cases from JSON file and structure them."""
    if not os.path.exists(ANALYSIS_FILE):
        return {}, [], "Analysis file not found"
    
    with open(ANALYSIS_FILE, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, dict) and 'results' in data:
        data = data['results']
        
    cases_by_class = {}
    all_categories = set()
    
    for item in data:
        if 'error_analysis' not in item:
            continue
            
        gt_class = item['ground_truth_label']
        category = item['error_analysis'].get('category', 'Unknown')
        
        if gt_class not in cases_by_class:
            cases_by_class[gt_class] = {}
            
        if category not in cases_by_class[gt_class]:
            cases_by_class[gt_class][category] = []
            
        # Format for dropdown
        case_label = f"ID {item['object_id']}: Pred '{item['gpt_cls_label']}'"
        cases_by_class[gt_class][category].append({
            'label': case_label,
            'value': item
        })
        all_categories.add(category)
        
    classes = sorted(list(cases_by_class.keys()))
    return cases_by_class, sorted(list(all_categories)), "Loaded cases successfully"

# Global storage for loaded cases to avoid reloading
BAD_CASES_DATA = None
BAD_CASES_CLASSES = []

def init_bad_cases():
    global BAD_CASES_DATA, BAD_CASES_CLASSES
    BAD_CASES_DATA, categories, msg = load_bad_cases()
    if BAD_CASES_DATA:
        BAD_CASES_CLASSES = sorted(list(BAD_CASES_DATA.keys()))
    return BAD_CASES_CLASSES

def update_error_types(class_name):
    if not BAD_CASES_DATA or class_name not in BAD_CASES_DATA:
        return gr.update(choices=[])
    
    types = sorted(list(BAD_CASES_DATA[class_name].keys()))
    return gr.update(choices=types, value=types[0] if types else None)

def update_case_list(class_name, error_type):
    if not BAD_CASES_DATA or class_name not in BAD_CASES_DATA or error_type not in BAD_CASES_DATA[class_name]:
        return gr.update(choices=[])
    
    cases = BAD_CASES_DATA[class_name][error_type]
    # Store the index in value for easy retrieval, or use object_id
    # Gradio dropdown values are strings. We can construct a string that holds the ID.
    choices = [c['label'] for c in cases]
    return gr.update(choices=choices, value=choices[0] if choices else None)

def load_selected_bad_case(class_name, error_type, case_label):
    if not BAD_CASES_DATA:
        return None, "Error: Data not loaded"
        
    try:
        cases = BAD_CASES_DATA[class_name][error_type]
        selected_case = next((c['value'] for c in cases if c['label'] == case_label), None)
        
        if not selected_case:
            return None, "Error: Case not found"
            
        # Load Point Cloud
        object_id = selected_case['object_id']
        
        # Reuse load_modelnet_pc logic but we need the index
        # object_id in analyse.json IS the modelnet index
        fig, _ = load_modelnet_pc(index=object_id, split='test', use_color=False)
        
        # Format Info Message
        info = f"""
### Case Details
- **Object ID / Index**: {object_id}
- **Ground Truth**: {selected_case['ground_truth_label']}
- **Prediction**: {selected_case['gpt_cls_label']}
- **Error Category**: {selected_case['error_analysis']['category']}

### Model Output
{selected_case['model_output']}

### GPT Reasoning
{selected_case['gpt_reason']}

### Error Analysis
{selected_case['error_analysis']['reasoning']}
        """
        return fig, info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error loading case: {e}"


def change_mode(mode):
    """根据模式切换显示不同的输入组件和可视化组件"""
    # 输出组件列表: 
    # [objaverse_inputs, glb_inputs, modelnet_inputs, pca_inputs, cluster_inputs, bad_case_inputs, plot_output, model3d_output, contour_outputs_group]
    
    # 默认隐藏所有输入组
    updates = [gr.update(visible=False)] * 9
    
    if mode == 'Objaverse':
        updates[0] = gr.update(visible=True)
        updates[6] = gr.update(visible=True)  # plot
        updates[7] = gr.update(visible=False) # model3d
    elif mode == 'GLB':
        updates[1] = gr.update(visible=True)
        updates[6] = gr.update(visible=False)
        updates[7] = gr.update(visible=True)
    elif mode == 'ModelNet40':
        updates[2] = gr.update(visible=True)
        updates[6] = gr.update(visible=True)
        updates[7] = gr.update(visible=False)
    elif mode == 'PCA点云':
        updates[3] = gr.update(visible=True)
        updates[6] = gr.update(visible=True)
        updates[7] = gr.update(visible=False)
    elif mode == 'Feature Clustering':
        updates[4] = gr.update(visible=True)
        updates[6] = gr.update(visible=True)
        updates[7] = gr.update(visible=False)
        updates[8] = gr.update(visible=True) # contour_outputs_group
    elif mode == 'Bad Case Analysis':
        updates[5] = gr.update(visible=True)
        updates[6] = gr.update(visible=True)
        updates[7] = gr.update(visible=False)
        
    return updates


def main():
    parser = argparse.ArgumentParser(description='点云和 3D 模型可视化 Gradio 应用')
    
    parser.add_argument('--port', type=int, default=7860,
                       help='Gradio 服务器端口（默认: 7860）')
    parser.add_argument('--server_name', type=str, default='0.0.0.0',
                       help='服务器地址（默认: 0.0.0.0）')
    parser.add_argument('--share', action='store_true',
                       help='是否创建公共链接')
    
    args = parser.parse_args()
    
    # Initialize bad cases data
    bad_case_classes = init_bad_cases()
    
    with gr.Blocks(title="点云可视化工具") as demo:
        gr.Markdown(
            """
            # 点云和 3D 模型可视化工具 🎨
            
            支持多种模式：
            1. **Objaverse 点云** - 通过对象 ID 加载点云
            2. **GLB 文件** - 输入服务器上的 GLB 文件路径
            3. **ModelNet40** - 浏览 ModelNet40 数据集
            4. **PCA点云** - 加载DINO特征PCA可视化点云
            5. **Feature Clustering** - 特征聚类分析
            6. **Bad Case Analysis** - 错误案例深度分析
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ['Objaverse', 'GLB', 'ModelNet40', 'PCA点云', 'Feature Clustering', 'Bad Case Analysis'],
                    value='Objaverse',
                    label='选择模式',
                    info='选择要可视化的点云类型'
                )
                
                # Objaverse 输入
                with gr.Group(visible=True) as objaverse_inputs:
                    objaverse_data_path = gr.Textbox(
                        label='数据路径',
                        placeholder='输入 Objaverse 数据路径',
                        value='data/objaverse_data'
                    )
                    objaverse_object_id = gr.Textbox(
                        label='对象 ID',
                        placeholder='输入对象 ID',
                        value=''
                    )
                    objaverse_pointnum = gr.Slider(
                        minimum=1024,
                        maximum=16384,
                        value=8192,
                        step=1024,
                        label='点数'
                    )
                    objaverse_use_color = gr.Checkbox(
                        label='使用颜色',
                        value=True
                    )
                    objaverse_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    objaverse_btn = gr.Button('加载 Objaverse 点云', variant='primary')
                
                # GLB 输入
                with gr.Group(visible=False) as glb_inputs:
                    glb_file_path = gr.Textbox(
                        label='GLB 文件路径',
                        placeholder='输入服务器上的 GLB 文件路径',
                        value=''
                    )
                    glb_btn = gr.Button('加载 GLB 文件', variant='primary')
                
                # ModelNet40 输入
                with gr.Group(visible=False) as modelnet_inputs:
                    modelnet_index = gr.Number(
                        label='数据索引',
                        value=0,
                        precision=0
                    )
                    modelnet_split = gr.Radio(
                        ['train', 'test'],
                        value='test',
                        label='数据集分割'
                    )
                    modelnet_use_color = gr.Checkbox(
                        label='使用颜色',
                        value=False
                    )
                    modelnet_config_path = gr.Textbox(
                        label='配置文件路径（可选）',
                        placeholder='留空使用默认路径',
                        value=''
                    )
                    modelnet_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    modelnet_btn = gr.Button('加载 ModelNet40 点云', variant='primary')
                
                # PCA点云输入
                with gr.Group(visible=False) as pca_inputs:
                    pca_file_path = gr.Textbox(
                        label='PCA点云文件路径 (.ply)',
                        placeholder='输入服务器上的 .ply 文件路径',
                        value=''
                    )
                    pca_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小（半径）'
                    )
                    pca_btn = gr.Button('加载 PCA 点云', variant='primary')
                
                # Feature Clustering 输入
                with gr.Group(visible=False) as cluster_inputs:
                    cluster_pc_path = gr.Textbox(
                        label='点云文件路径 (.npy)',
                        placeholder='输入 .npy 文件路径 (例如: .../xxx_8192.npy)',
                        value='/mnt/extra/Point-R1/example_material/npys/e85ebb729b02402bbe3b917e1196f8d3_8192.npy'
                    )
                    cluster_feat_path = gr.Textbox(
                        label='特征文件路径 (.npy)',
                        placeholder='输入 .npy 文件路径 (例如: .../xxx_features.npy)',
                        value='/mnt/extra/Point-R1/example_material/dino_features/e85ebb729b02402bbe3b917e1196f8d3_features.npy'
                    )
                    
                    with gr.Row():
                        cluster_alpha = gr.Slider(minimum=0.0001, maximum=0.2, value=0.04, step=0.0001, label='Alpha (空间约束强弱)')
                        
                    gr.Markdown("Beta (特征相似度阈值) - 4个层级")
                    with gr.Row():
                        cluster_beta1 = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label='Beta 1 (较松)')
                        cluster_beta2 = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label='Beta 2')
                    with gr.Row():
                        cluster_beta3 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label='Beta 3')
                        cluster_beta4 = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label='Beta 4 (较严)')
                        
                    cluster_point_size = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=1.5,
                        step=0.5,
                        label='点大小'
                    )
                    
                    cluster_calc_btn = gr.Button('计算聚类', variant='primary')
                    
                    # State 用于存储聚类结果
                    cluster_state = gr.State(None)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 聚类结果可视化与 Prompt 构建")
                    
                    cluster_glb_path = gr.Textbox(
                        label='GLB 文件路径 (用于渲染)',
                                value='/mnt/extra/Point-R1/example_material/glbs/e85ebb729b02402bbe3b917e1196f8d3.glb'
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("1. 单簇浏览 (Old)"):
                            cluster_level = gr.Radio(
                                ['层级 1', '层级 2', '层级 3', '层级 4'],
                                value='层级 1',
                                label='显示层级'
                            )
                            cluster_id_input = gr.Number(label='簇 ID (-1 为自动最大)', value=-1, precision=0)
                            
                            cam_azimuth = gr.Slider(0, 360, value=0, label="Azimuth")
                            cam_elevation = gr.Slider(-90, 90, value=0, label="Elevation")
                            cam_radius = gr.Slider(0.5, 3.0, value=1.5, label="Radius")
                            ureca_style_checkbox = gr.Checkbox(label="URECA Style", value=True)
                            
                            render_contour_btn = gr.Button("渲染单簇轮廓", variant="secondary")
                        
                        with gr.Tab("2. 兄弟簇 SoM 构建 (New)"):
                            gr.Markdown("选择一个父节点，系统将自动识别其所有子节点（兄弟簇），并在4个最佳视角下生成 SoM 视觉提示图。")
                            
                            som_parent_level = gr.Radio(
                                ['层级 1', '层级 2', '层级 3', '层级 4'],
                                value='层级 1',
                                label='父节点层级'
                            )
                            som_parent_id = gr.Number(label='父节点 ID', value=0, precision=0)
                            
                            som_gen_btn = gr.Button("生成兄弟簇 SoM 组图", variant="primary")
                                
                # Bad Case Analysis Inputs
                with gr.Group(visible=False) as bad_case_inputs:
                    gr.Markdown("### Bad Case Filter")
                    bc_class = gr.Dropdown(choices=bad_case_classes, label="Target Class", allow_custom_value=True)
                    bc_type = gr.Dropdown(choices=[], label="Error Type", allow_custom_value=True)
                    bc_case = gr.Dropdown(choices=[], label="Select Case", allow_custom_value=True)
                    bc_btn = gr.Button("Visualize Bad Case", variant="primary")

                info_output = gr.Markdown(label='Info')
            
            with gr.Column(scale=2):
                plot_output = gr.Plot(label='点云可视化', visible=True)
                model3d_output = gr.Model3D(label='3D 模型可视化', visible=False)
                
                # Feature Clustering Outputs
                with gr.Group(visible=False) as contour_outputs_group:
                    
                    # 单簇输出区
                    with gr.Group(visible=True) as single_cluster_outputs:
                        gr.Markdown("### 单簇视图")
                    with gr.Row():
                            contour_output_context = gr.Image(label="Context", type="numpy")
                            contour_output_local = gr.Image(label="Local", type="numpy")
                    
                    # SoM 输出区
                    with gr.Group(visible=True) as som_outputs:
                        gr.Markdown("### 兄弟簇 SoM 组图 (6 Views)")
                    with gr.Row():
                        som_v1_l = gr.Image(label="View 1 (Clean)", type="numpy")
                        som_v1_r = gr.Image(label="View 1 (SoM)", type="numpy")
                    with gr.Row():
                        som_v2_l = gr.Image(label="View 2 (Clean)", type="numpy")
                        som_v2_r = gr.Image(label="View 2 (SoM)", type="numpy")
                    with gr.Row():
                        som_v3_l = gr.Image(label="View 3 (Clean)", type="numpy")
                        som_v3_r = gr.Image(label="View 3 (SoM)", type="numpy")
                    with gr.Row():
                        som_v4_l = gr.Image(label="View 4 (Clean)", type="numpy")
                        som_v4_r = gr.Image(label="View 4 (SoM)", type="numpy")
                    with gr.Row():
                        som_v5_l = gr.Image(label="View 5 (Clean)", type="numpy")
                        som_v5_r = gr.Image(label="View 5 (SoM)", type="numpy")
                    with gr.Row():
                        som_v6_l = gr.Image(label="View 6 (Clean)", type="numpy")
                        som_v6_r = gr.Image(label="View 6 (SoM)", type="numpy")
        
        # 模式切换
        mode.change(
            change_mode,
            inputs=[mode],
            outputs=[objaverse_inputs, glb_inputs, modelnet_inputs, pca_inputs, cluster_inputs, bad_case_inputs, plot_output, model3d_output, contour_outputs_group]
        )
        
        # Objaverse 按钮事件
        objaverse_btn.click(
            load_objaverse_pc,
            inputs=[objaverse_data_path, objaverse_object_id, objaverse_pointnum, objaverse_use_color, objaverse_point_size],
            outputs=[plot_output, info_output]
        )
        
        # GLB 按钮事件
        def load_glb_wrapper(file_path):
            if file_path is None or file_path.strip() == "":
                return None, "错误: 请输入 GLB 文件路径"
            file_path = file_path.strip()
            if not os.path.exists(file_path):
                return None, f"错误: 文件不存在: {file_path}"
            if not file_path.lower().endswith('.glb'):
                return None, "错误: 文件必须是 .glb 格式"
            return load_glb_model(file_path)
        
        glb_btn.click(
            load_glb_wrapper,
            inputs=[glb_file_path],
            outputs=[model3d_output, info_output]
        )
        
        # ModelNet40 按钮事件
        modelnet_btn.click(
            load_modelnet_pc,
            inputs=[modelnet_index, modelnet_split, modelnet_use_color, modelnet_config_path, modelnet_point_size],
            outputs=[plot_output, info_output]
        )
        
        # PCA点云按钮事件
        def load_pca_wrapper(file_path, point_size):
            if file_path is None or file_path.strip() == "":
                return None, "错误: 请输入 PCA 点云文件路径"
            file_path = file_path.strip()
            return load_pca_pointcloud(file_path, point_size)
        
        pca_btn.click(
            load_pca_wrapper,
            inputs=[pca_file_path, pca_point_size],
            outputs=[plot_output, info_output]
        )
        
        # Feature Clustering 事件
        cluster_calc_btn.click(
            load_and_cluster_points,
            inputs=[
                cluster_pc_path, cluster_feat_path, 
                cluster_alpha, 
                cluster_beta1, cluster_beta2, cluster_beta3, cluster_beta4,
                cluster_point_size
            ],
            outputs=[plot_output, cluster_state, info_output]
        )
        
        # Old single cluster interaction
        cluster_level.change(
            update_cluster_view,
            inputs=[cluster_state, cluster_level],
            outputs=[plot_output, info_output]
        )
        
        render_contour_btn.click(
            render_cluster_contour,
            inputs=[
                cluster_state, cluster_glb_path, cluster_id_input, 
                cluster_level, cam_azimuth, cam_elevation, cam_radius,
                ureca_style_checkbox
            ],
            outputs=[contour_output_context, contour_output_local, info_output]
        )
        
        # New SoM Interaction
        som_gen_btn.click(
            render_sibling_group_views,
            inputs=[cluster_state, cluster_glb_path, som_parent_level, som_parent_id],
            outputs=[
                som_v1_l, som_v1_r,
                som_v2_l, som_v2_r,
                som_v3_l, som_v3_r,
                som_v4_l, som_v4_r,
                som_v5_l, som_v5_r,
                som_v6_l, som_v6_r,
                info_output
            ]
        )
        
        # Bad Case Events
        bc_class.change(update_error_types, inputs=[bc_class], outputs=[bc_type])
        bc_type.change(update_case_list, inputs=[bc_class, bc_type], outputs=[bc_case])
        bc_btn.click(
            load_selected_bad_case,
            inputs=[bc_class, bc_type, bc_case],
            outputs=[plot_output, info_output]
        )
        
        gr.Markdown(
            """
            ### 使用说明：
            - **Objaverse 模式**: 输入数据路径和对象 ID，点击加载按钮（点云可视化）
            - **GLB 模式**: 输入服务器上的 GLB 文件路径，点击加载按钮（3D 模型可视化）
            - **ModelNet40 模式**: 输入数据索引和数据集分割，点击加载按钮（点云可视化）
            - **PCA点云模式**: 输入 .ply 文件路径（例如：`example_material/dino_features/xxx_pca.ply`），点击加载按钮（点云可视化）
            - **Feature Clustering**: 输入点云(.npy)和特征(.npy)路径，设置距离阈值Alpha和4个相似度阈值Beta，点击计算。计算完成后可切换层级查看聚类结果。
            - **Bad Case Analysis**: 选择类别和错误类型，查看具体的错误案例和模型推理过程。
            
            ### 提示：
            - 点云可视化支持鼠标交互（旋转、缩放、平移）
            - 3D 模型可视化支持完整的 3D 交互（旋转、缩放、平移、材质查看）
            - 如果点云没有颜色信息，将显示为灰色
            - PCA点云的颜色表示特征的PCA降维结果（RGB对应前3个主成分）
            """
        )
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        allowed_paths=['/mnt/extra/tmp']
    )


if __name__ == '__main__':
    main()
