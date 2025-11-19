"""
从多视角图像提取DINO特征并反投影到点云，然后进行PCA可视化
参考back-to-3d项目的实现
支持运行时渲染和精确的可见性检测
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.decomposition import PCA
import open3d as o3d
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入运行时渲染模块
from renderer_o3d import (
    Open3DRenderer,
    check_visible_points_with_depth,
    project_points_to_image_with_depth,
    sample_view_points as o3d_sample_view_points,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint
)

# 导入transformers用于加载DINOv3模型（参考pca.ipynb）
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms.functional as TF


def adjust_image_size_for_patch(image_size, patch_size=16):
    """
    调整图像尺寸使其是patch_size的倍数（DINO要求）
    
    参数:
        image_size: 原始图像尺寸
        patch_size: patch尺寸（默认16，DINOv3使用16）
    
    返回:
        调整后的图像尺寸（patch_size的倍数）
    """
    # 向下取整到最近的patch_size倍数
    adjusted_size = (image_size // patch_size) * patch_size
    if adjusted_size < patch_size:
        adjusted_size = patch_size
    return adjusted_size


def load_pointcloud(npy_path):
    """加载点云数据"""
    points = np.load(npy_path)
    # 假设点云格式是 (N, 6) 或 (N, 3)，只取xyz坐标
    if points.shape[1] >= 3:
        return points[:, :3].astype(np.float32)
    else:
        return points.astype(np.float32)


def load_rendered_images(render_dir):
    """
    加载预渲染的多视角图像（向后兼容）
    
    参数:
        render_dir: 渲染图像目录
    
    返回:
        images: (V, H, W, 3) numpy数组
    """
    render_dir = Path(render_dir)
    image_files = sorted(render_dir.glob("view_*.png"))
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        images.append(np.array(img))
    return np.array(images)  # (V, H, W, 3)


# 使用renderer_o3d中的函数
project_points_to_image = project_points_to_image_with_depth


def get_feature_for_pixel_location(feature_map, pixel_locations, image_size=1024, patch_size=None):
    """
    根据像素坐标获取对应的特征（参考back-to-3d的实现）
    
    参数:
        feature_map: (num_patches, emb_dim) 特征图
        pixel_locations: (N, 2) 像素坐标 [x, y]
        image_size: 图像尺寸
        patch_size: patch尺寸（如果为None，则从feature_map自动计算）
    
    返回:
        features: (N, emb_dim) 每个点对应的特征
    """
    # 动态计算patch_size（参考back-to-3d的实现）
    if patch_size is None:
        num_patches = feature_map.shape[0]
        num_patches_per_side = int(np.sqrt(num_patches))
        patch_size = image_size / num_patches_per_side
    
    # 计算patch ID（参考back-to-3d的实现）
    # patch_id = y // patch_size * (image_size / patch_size) + x // patch_size
    patch_id_y = (pixel_locations[:, 1] // patch_size).astype(int)
    patch_id_x = (pixel_locations[:, 0] // patch_size).astype(int)
    num_patches_per_side = int(image_size / patch_size)
    patch_id = patch_id_y * num_patches_per_side + patch_id_x
    
    # 确保patch_id在有效范围内
    patch_id = np.clip(patch_id, 0, feature_map.shape[0] - 1)
    
    # 获取对应的特征
    features = feature_map[patch_id]
    return features


def extract_single_image_feature(img_array, model, torch_device, view_idx=None):
    """
    提取单张图像的特征（用于并行处理）
    
    参数:
        img_array: (H, W, 3) numpy数组，RGB图像，值域[0, 255]
        model: DINOv3模型（transformers AutoModel）
        torch_device: torch.device对象
        view_idx: 视角索引（用于显示进度）
    
    返回:
        feature: (num_patches, emb_dim) 特征
    """
    # ImageNet归一化参数（参考pca.ipynb）
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    # 转换为PIL Image
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    
    # 转换为tensor并归一化到[0, 1]
    img_tensor = TF.to_tensor(img_pil)  # (3, H, W)，值域[0, 1]
    
    # ImageNet归一化（参考pca.ipynb）
    img_normalized = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img_batch = img_normalized.unsqueeze(0).to(torch_device)  # (1, 3, H, W)
    
    # 前向传播（参考pca.ipynb Cell 10）
    with torch.inference_mode():
        # torch.autocast需要字符串类型的device_type
        device_type_str = 'cuda' if torch_device.type == 'cuda' else 'cpu'
        
        with torch.autocast(device_type=device_type_str, dtype=torch.float32):
            output = model(img_batch)
            hidden = output.last_hidden_state  # (1, N_tokens, C)
            
            # 跳过特殊token（CLS token + register tokens）
            num_register = getattr(model.config, "num_register_tokens", 0)
            num_special = 1 + num_register
            x = hidden[:, num_special:, :]  # (1, N_patches, C)
            x = x.squeeze(0).detach().cpu()  # (N_patches, C)
    
    return x.numpy()


def extract_dino_features_from_images(images, model=None, models=None, devices=None, image_processor=None, num_gpus=4):
    """
    从多视角图像提取DINO特征（参考pca.ipynb的前向传播方式）
    支持多GPU并行处理
    
    参数:
        images: (V, H, W, 3) numpy数组，RGB图像，值域[0, 255]
        model: DINOv3模型（transformers AutoModel）- 单GPU模式
        models: 模型列表（多GPU模式）- 每个GPU一个模型实例
        devices: 设备列表（多GPU模式）
        image_processor: AutoImageProcessor（可选）
        num_gpus: 使用的GPU数量（默认4）
    
    返回:
        features: (V, num_patches, emb_dim) 每个视角的特征
    """
    num_views = images.shape[0]
    
    # 多GPU模式：使用多个模型实例并行处理
    if models is not None and devices is not None:
        print(f"  使用多GPU并行处理（{len(devices)}个GPU）...")
        features_list = [None] * num_views
        
        # 将视角分配到不同的GPU
        def process_batch(view_indices, gpu_idx):
            """处理一批视角"""
            batch_features = []
            for view_idx in view_indices:
                feature = extract_single_image_feature(
                    images[view_idx], 
                    models[gpu_idx], 
                    devices[gpu_idx],
                    view_idx
                )
                batch_features.append((view_idx, feature))
            return batch_features
        
        # 分配视角到不同GPU
        views_per_gpu = num_views // len(devices)
        remainder = num_views % len(devices)
        
        tasks = []
        start_idx = 0
        for gpu_idx in range(len(devices)):
            end_idx = start_idx + views_per_gpu + (1 if gpu_idx < remainder else 0)
            view_indices = list(range(start_idx, end_idx))
            if len(view_indices) > 0:
                tasks.append((view_indices, gpu_idx))
            start_idx = end_idx
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {executor.submit(process_batch, view_indices, gpu_idx): gpu_idx 
                      for view_indices, gpu_idx in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取DINO特征"):
                batch_features = future.result()
                for view_idx, feature in batch_features:
                    features_list[view_idx] = feature
        
        features = np.array(features_list)  # (V, num_patches, emb_dim)
        return features
    
    # 单GPU模式：使用单个模型顺序处理
    else:
        if model is None:
            raise ValueError("单GPU模式需要提供model参数")
        
        # 确保device是torch.device对象
        if isinstance(model, torch.nn.Module):
            torch_device = next(model.parameters()).device
        else:
            torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        features_list = []
        for view_idx in tqdm(range(num_views), desc="提取DINO特征"):
            feature = extract_single_image_feature(
                images[view_idx], 
                model, 
                torch_device,
                view_idx
            )
            features_list.append(feature)
        
        features = np.array(features_list)  # (V, num_patches, emb_dim)
        return features


def backproject_features_to_points(
    points, 
    images, 
    camera_intrinsics, 
    camera_extrinsics, 
    dino_features, 
    image_size=None,
    patch_size=16,
    only_visible=True,
    depth_maps=None,
    depth_threshold=0.01
):
    """
    将DINO特征反投影到3D点（改进版：支持深度图可见性检测）
    
    参数:
        points: (N, 3) 3D点坐标
        images: (V, H, W, 3) 渲染图像 或 list of (H, W, 3)
        camera_intrinsics: list of Open3D相机内参对象
        camera_extrinsics: list of (4, 4) 相机外参矩阵
        dino_features: (V, num_patches, emb_dim) DINO特征
        image_size: 图像尺寸（如果为None，则从images自动检测）
        patch_size: patch尺寸
        only_visible: 是否只使用可见点的特征
        depth_maps: list of (H, W) 深度图（如果提供，用于精确可见性检测）
        depth_threshold: 深度阈值（用于可见性检测）
    
    返回:
        point_features: (N, emb_dim) 每个点的特征
        visibility_count: (N,) 每个点被看到的次数
    """
    num_points = points.shape[0]
    num_views = len(images)
    emb_dim = dino_features.shape[2]
    
    # 自动检测图像尺寸
    if image_size is None:
        image_size = images.shape[1]  # 假设是正方形图像
    
    # 动态计算patch_size（参考back-to-3d的实现）
    # 如果patch_size未指定，从特征图自动计算
    if patch_size is None or patch_size == 16:
        # 根据实际的特征图大小计算patch_size
        actual_num_patches = dino_features.shape[1]
        num_patches_per_side = int(np.sqrt(actual_num_patches))
        if num_patches_per_side > 0:
            calculated_patch_size = image_size / num_patches_per_side
            if abs(calculated_patch_size - patch_size) > 0.1:  # 如果差异较大
                print(f"  动态计算patch_size: {calculated_patch_size:.2f} (原值: {patch_size})")
                patch_size = calculated_patch_size
    
    # 初始化特征累加器
    point_features = np.zeros((num_points, emb_dim), dtype=np.float32)
    visibility_count = np.zeros(num_points, dtype=np.float32)
    
    # 对每个视角进行处理
    for view_idx in range(num_views):
        # 投影3D点到图像平面
        pixel_coords, depths, valid_mask = project_points_to_image(
            points, 
            camera_intrinsics[view_idx], 
            camera_extrinsics[view_idx],
            image_size=image_size
        )
        
        if only_visible:
            # 使用深度图进行精确可见性检测（如果可用）
            if depth_maps is not None and view_idx < len(depth_maps) and depth_maps[view_idx] is not None:
                # 使用深度图检查可见性（参考back-to-3d的精确检测）
                # 添加调试信息：打印深度分布（只打印前若干个视角以避免日志过多）
                if view_idx < 3:
                    valid_depths = depths[valid_mask]
                    depth_map_valid = depth_maps[view_idx][depth_maps[view_idx] > 0]
                    if len(valid_depths) > 0 and len(depth_map_valid) > 0:
                        d_min, d_max = valid_depths.min(), valid_depths.max()
                        d_mean = valid_depths.mean()
                        d_p5, d_p50, d_p95 = np.percentile(valid_depths, [5, 50, 95])
                        m_min, m_max = depth_map_valid.min(), depth_map_valid.max()
                        m_mean = depth_map_valid.mean()
                        m_p5, m_p50, m_p95 = np.percentile(depth_map_valid, [5, 50, 95])
                        print(
                            f"    视角 {view_idx}: 点深度分布 -> "
                            f"min={d_min:.4f}, max={d_max:.4f}, mean={d_mean:.4f}, "
                            f"p5={d_p5:.4f}, p50={d_p50:.4f}, p95={d_p95:.4f}"
                        )
                        print(
                            f"    视角 {view_idx}: 深度图分布 -> "
                            f"min={m_min:.4f}, max={m_max:.4f}, mean={m_mean:.4f}, "
                            f"p5={m_p5:.4f}, p50={m_p50:.4f}, p95={m_p95:.4f}"
                        )
                        print(
                            f"    当前深度可见性阈值比例(relative_threshold_ratio)=0.05, "
                            f"depth_threshold参数={depth_threshold}"
                        )
                
                depth_visible = check_visible_points_with_depth(
                    points,
                    pixel_coords,
                    depths,
                    depth_maps[view_idx],
                    depth_threshold=depth_threshold,
                    use_relative_threshold=True,  # 使用相对阈值（5%）
                    relative_threshold_ratio=0.1
                )
                # 结合基本可见性检查和深度图检查
                valid_indices = np.where(valid_mask & depth_visible)[0]
            else:
                # 只使用基本的可见性检查
                valid_indices = np.where(valid_mask)[0]
        else:
            valid_indices = np.arange(num_points)
        
        if len(valid_indices) == 0:
            continue
        
        # 获取这些点对应的特征（动态计算patch_size）
        valid_pixel_coords = pixel_coords[valid_indices]
        # 根据当前视角的特征图动态计算patch_size
        actual_num_patches = dino_features[view_idx].shape[0]
        num_patches_per_side = int(np.sqrt(actual_num_patches))
        dynamic_patch_size = image_size / num_patches_per_side if num_patches_per_side > 0 else patch_size
        
        view_features = get_feature_for_pixel_location(
            dino_features[view_idx],
            valid_pixel_coords,
            image_size=image_size,
            patch_size=dynamic_patch_size
        )
        
        # 累加特征
        point_features[valid_indices] += view_features
        visibility_count[valid_indices] += 1
    
    # 平均化特征（只对可见点）
    visible_mask = visibility_count > 0
    point_features[visible_mask] /= visibility_count[visible_mask, None]
    
    return point_features, visibility_count


def save_camera_params(render_dir, viewpoints, output_path):
    """
    保存相机参数（用于后续反投影）
    
    参数:
        render_dir: 渲染图像目录
        viewpoints: (V, 3) 视角点位置
        output_path: 输出JSON文件路径
    """
    # 从render_open3d_offscreen.py我们知道相机参数
    # 内参：1024x1024, fov=60度
    width = height = 1024
    fov = 60.0
    fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
    cx = cy = width / 2.0
    
    camera_params = {
        'intrinsic': {
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'fov': fov
        },
        'viewpoints': viewpoints.tolist(),
        'num_views': len(viewpoints)
    }
    
    with open(output_path, 'w') as f:
        json.dump(camera_params, f, indent=2)


def load_camera_params(json_path):
    """加载相机参数"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params




def visualize_features_with_pca(points, features, output_path=None):
    """
    使用PCA将特征降维到3D并可视化
    
    参数:
        points: (N, 3) 点云坐标
        features: (N, emb_dim) 特征
        output_path: 输出点云文件路径（可选）
    """
    num_points = len(points)
    
    # 识别有效特征点（非NaN且特征和不为0）
    valid_mask = ~np.isnan(features).any(axis=1) & (features.sum(axis=1) != 0)
    valid_features = features[valid_mask]
    num_valid = valid_mask.sum()
    
    print(f"  总点数: {num_points}, 有效特征点数: {num_valid}, 无效点数: {num_points - num_valid}")
    
    if len(valid_features) == 0:
        print("警告：没有有效的特征点")
        return None
    
    # 只对有效特征进行PCA
    pca = PCA(n_components=3)
    features_pca_valid = pca.fit_transform(valid_features)
    
    # 归一化到[0, 1]
    features_pca_valid = (features_pca_valid - features_pca_valid.min(axis=0)) / (
        features_pca_valid.max(axis=0) - features_pca_valid.min(axis=0) + 1e-8
    )
    
    # 为所有点创建颜色数组
    colors_all = np.ones((num_points, 3), dtype=np.float32) * 0.3  # 默认灰色（不可见点）
    colors_all[valid_mask] = features_pca_valid  # 有效点使用PCA颜色
    
    # 创建点云（包含所有点）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"保存可视化点云到: {output_path} (包含所有 {num_points} 个点)")
    
    return pcd


def visualize_depth_map(depth_map, output_path=None):
    """
    可视化深度图并保存
    
    参数:
        depth_map: (H, W) 深度图
        output_path: 输出图像路径（可选）
    
    返回:
        depth_vis: (H, W, 3) RGB可视化图像
    """
    # 归一化深度值到[0, 1]
    depth_valid = depth_map[depth_map > 0]
    if len(depth_valid) == 0:
        # 如果没有有效深度值，返回黑色图像
        depth_vis = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
    else:
        depth_min = depth_valid.min()
        depth_max = depth_valid.max()
        if depth_max > depth_min:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)
        
        # 使用colormap转换为RGB
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_vis = cm.viridis(depth_norm)[:, :, :3]  # (H, W, 3)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        
        # 背景设为黑色
        depth_vis[depth_map <= 0] = 0
    
    if output_path:
        Image.fromarray(depth_vis).save(output_path)
    
    return depth_vis


def visualize_dino_features_pca_2d(dino_features, intermediate_dir, image_size=None, patch_size=14):
    """
    对每个视角的DINO特征进行PCA降维并可视化保存为2D图像
    
    参数:
        dino_features: (V, num_patches, emb_dim) DINO特征
        intermediate_dir: 中间结果目录
        image_size: 图像尺寸（用于reshape特征图）
        patch_size: patch尺寸
    """
    num_views = dino_features.shape[0]
    num_patches = dino_features.shape[1]
    emb_dim = dino_features.shape[2]
    
    # 计算特征图的尺寸
    if image_size is None:
        num_patches_per_side = int(np.sqrt(num_patches))
        feature_map_size = num_patches_per_side * patch_size
    else:
        feature_map_size = image_size
        num_patches_per_side = int(np.sqrt(num_patches))
    
    # 创建输出目录
    pca_2d_dir = os.path.join(intermediate_dir, "dino_features_pca")
    os.makedirs(pca_2d_dir, exist_ok=True)
    
    # 对所有视角的特征进行PCA（使用全局PCA以保持一致性）
    all_features = dino_features.reshape(-1, emb_dim)  # (V*num_patches, emb_dim)
    pca = PCA(n_components=3)
    all_features_pca = pca.fit_transform(all_features)  # (V*num_patches, 3)
    
    # 归一化到[0, 1]
    all_features_pca = (all_features_pca - all_features_pca.min(axis=0)) / (
        all_features_pca.max(axis=0) - all_features_pca.min(axis=0) + 1e-8
    )
    
    # 对每个视角进行可视化
    for view_idx in range(num_views):
        # 获取当前视角的特征PCA
        view_features_pca = all_features_pca[view_idx * num_patches:(view_idx + 1) * num_patches]  # (num_patches, 3)
        
        # Reshape到特征图尺寸
        feature_map_rgb = view_features_pca.reshape(num_patches_per_side, num_patches_per_side, 3)
        
        # 如果特征图尺寸小于目标图像尺寸，进行上采样
        if feature_map_size != num_patches_per_side:
            # 使用PIL进行高质量上采样
            feature_map_img = Image.fromarray((feature_map_rgb * 255).astype(np.uint8))
            feature_map_img = feature_map_img.resize((feature_map_size, feature_map_size), Image.Resampling.LANCZOS)
            feature_map_rgb = np.array(feature_map_img) / 255.0
        
        # 转换为uint8并保存
        feature_map_rgb_uint8 = (feature_map_rgb * 255).astype(np.uint8)
        output_path = os.path.join(pca_2d_dir, f"view_{view_idx:03d}_pca.png")
        Image.fromarray(feature_map_rgb_uint8).save(output_path)
    
    print(f"  保存DINO特征PCA可视化到: {pca_2d_dir} ({num_views}个视角)")


def save_2d_intermediate_results(
    images,
    depth_maps,
    dino_features,
    output_dir,
    object_id,
    points,
    camera_intrinsics,
    camera_extrinsics,
    image_size=None,
    patch_size=14
):
    """
    保存2D中间结果：渲染图、深度图、DINO特征PCA可视化
    
    参数:
        images: (V, H, W, 3) RGB渲染图像
        depth_maps: list of (H, W) 深度图（如果为None则跳过）
        dino_features: (V, num_patches, emb_dim) DINO特征
        output_dir: 输出目录
        object_id: 对象ID
        image_size: 图像尺寸
        patch_size: patch尺寸
    """
    print("  保存2D中间结果...")
    
    # 创建输出目录
    intermediate_dir = os.path.join(output_dir, f"{object_id}_intermediate_2d")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # 1. 保存渲染图像
    render_dir = os.path.join(intermediate_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    num_views = images.shape[0]
    for view_idx in range(num_views):
        render_path = os.path.join(render_dir, f"view_{view_idx:03d}.png")
        Image.fromarray(images[view_idx].astype(np.uint8)).save(render_path)
    print(f"    保存渲染图像到: {render_dir} ({num_views}个视角)")
    
    # 2. 保存GLB渲染深度图 & 计算点云深度图 + 差异可视化 & 点云RGB图
    if depth_maps is not None and len(depth_maps) > 0:
        depth_dir = os.path.join(intermediate_dir, "depth_maps")
        pc_depth_dir = os.path.join(intermediate_dir, "pc_depth_maps")
        diff_depth_dir = os.path.join(intermediate_dir, "depth_diff_overlays")
        pc_rgb_dir = os.path.join(intermediate_dir, "pc_renders")
        pc_on_glb_dir = os.path.join(intermediate_dir, "pc_on_glb_overlays")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(pc_depth_dir, exist_ok=True)
        os.makedirs(diff_depth_dir, exist_ok=True)
        os.makedirs(pc_rgb_dir, exist_ok=True)
        os.makedirs(pc_on_glb_dir, exist_ok=True)

        num_views = len(depth_maps)
        for view_idx in range(num_views):
            glb_depth = depth_maps[view_idx]
            if glb_depth is None:
                continue

            # 2.1 保存 GLB 深度图
            depth_path = os.path.join(depth_dir, f"view_{view_idx:03d}_depth.png")
            glb_depth_vis = visualize_depth_map(glb_depth, depth_path)

            # 2.2 计算点云深度图（在同一视角、同一相机参数下）
            H, W = glb_depth.shape
            pc_depth = np.zeros_like(glb_depth, dtype=np.float32)

            pixel_coords, depths, valid_mask = project_points_to_image(
                points,
                camera_intrinsics[view_idx],
                camera_extrinsics[view_idx],
                image_size=H,
            )

            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size > 0:
                xs = pixel_coords[valid_indices, 0]
                ys = pixel_coords[valid_indices, 1]
                ds = depths[valid_indices]

                xs_int = np.clip(xs.astype(int), 0, W - 1)
                ys_int = np.clip(ys.astype(int), 0, H - 1)

                # 构建点云深度图：同一像素取最近点深度
                for x_i, y_i, d_i in zip(xs_int, ys_int, ds):
                    prev = pc_depth[y_i, x_i]
                    if prev == 0 or d_i < prev:
                        pc_depth[y_i, x_i] = d_i

                # 额外：渲染点云在当前视角下的 RGB 图（黑底 + 绿色点）
                pc_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                pc_rgb[ys_int, xs_int] = np.array([0, 255, 0], dtype=np.uint8)
                pc_rgb_path = os.path.join(pc_rgb_dir, f"view_{view_idx:03d}_pc_rgb.png")
                Image.fromarray(pc_rgb).save(pc_rgb_path)

                # 额外：将点云（红色）叠加到 GLB 渲染图上，便于检查视角是否一致
                glb_rgb = images[view_idx].astype(np.uint8).copy()
                glb_rgb[ys_int, xs_int] = np.array([255, 0, 0], dtype=np.uint8)
                pc_on_glb_path = os.path.join(
                    pc_on_glb_dir, f"view_{view_idx:03d}_overlay_pc_on_glb.png"
                )
                Image.fromarray(glb_rgb).save(pc_on_glb_path)

            # 保存点云深度图
            pc_depth_path = os.path.join(pc_depth_dir, f"view_{view_idx:03d}_pc_depth.png")
            pc_depth_vis = visualize_depth_map(pc_depth, pc_depth_path)

            # 2.3 计算 GLB 深度图与点云深度图的差异，并叠加可视化
            valid_both = (glb_depth > 0) & (pc_depth > 0)
            if not np.any(valid_both):
                continue

            diff = np.zeros_like(glb_depth, dtype=np.float32)
            diff[valid_both] = pc_depth[valid_both] - glb_depth[valid_both]

            max_abs_diff = np.max(np.abs(diff[valid_both]))
            if max_abs_diff < 1e-8:
                # 差异几乎为 0，直接使用 GLB 深度可视化
                overlay_img = glb_depth_vis
            else:
                # 将差异映射到 [-1, 1] 再映射到 [0, 1] 用于 colormap
                norm_diff = diff / (2.0 * max_abs_diff) + 0.5
                norm_diff = np.clip(norm_diff, 0.0, 1.0)
                diff_color = cm.seismic(norm_diff)[:, :, :3]  # (H, W, 3), 0~1
                diff_color = (diff_color * 255).astype(np.uint8)

                # 将差异着色图叠加到 GLB 深度可视化上
                alpha = 0.6
                overlay_img = (
                    (1.0 - alpha) * glb_depth_vis.astype(np.float32)
                    + alpha * diff_color.astype(np.float32)
                )
                overlay_img = overlay_img.astype(np.uint8)

            diff_overlay_path = os.path.join(
                diff_depth_dir, f"view_{view_idx:03d}_depth_with_pc_diff.png"
            )
            Image.fromarray(overlay_img).save(diff_overlay_path)

        print(f"    保存GLB深度图到: {depth_dir} ({num_views}个视角)")
        print(f"    保存点云深度图到: {pc_depth_dir} ({num_views}个视角)")
        print(f"    保存深度差异叠加图到: {diff_depth_dir} ({num_views}个视角)")
        print(f"    保存点云RGB图到: {pc_rgb_dir} ({num_views}个视角，部分视角可能为空)")
        print(f"    保存点云叠加GLB图到: {pc_on_glb_dir} ({num_views}个视角，部分视角可能为空)")
    
    # 3. 保存DINO特征PCA可视化
    visualize_dino_features_pca_2d(
        dino_features,
        intermediate_dir,
        image_size=image_size,
        patch_size=patch_size
    )


def process_single_object(
    object_id,
    npy_dir='example_material/npys',
    glb_dir='example_material/glbs',
    render_dir='example_material/renders_o3d',
    output_dir='example_material/dino_features',
    device='cuda',
    image_size=1024,
    patch_size=16,
    partition=5,
    radius=1.1,
    use_runtime_rendering=True,
    depth_threshold=0.01
):
    """
    处理单个对象：提取DINO特征、反投影、PCA可视化
    
    参数:
        object_id: 对象ID（不含扩展名）
        npy_dir: 点云目录
        glb_dir: GLB文件目录
        render_dir: 渲染图像目录（如果use_runtime_rendering=False）
        output_dir: 输出目录
        device: 设备
        image_size: 图像尺寸（会自动调整为patch_size的倍数）
        patch_size: patch尺寸
        partition: 视角细分级别
        radius: 相机距离半径
        use_runtime_rendering: 是否使用运行时渲染（True：实时渲染，False：使用预渲染图像）
        depth_threshold: 深度阈值（用于可见性检测）
    """
    print(f"\n处理对象: {object_id}")
    
    # 调整图像尺寸使其是patch_size的倍数（DINO要求）
    original_image_size = image_size
    image_size = adjust_image_size_for_patch(image_size, patch_size)
    if image_size != original_image_size:
        print(f"  调整图像尺寸: {original_image_size} -> {image_size} (patch_size={patch_size}的倍数)")
    
    # 1. 加载点云
    npy_path = os.path.join(npy_dir, f"{object_id}_8192.npy")
    if not os.path.exists(npy_path):
        print(f"错误：找不到点云文件 {npy_path}")
        return
    
    points = load_pointcloud(npy_path)
    print(f"  加载点云: {points.shape}")

    # 对点云做坐标轴对齐 + 自身归一化，使其与mesh坐标系一致
    # 原始点云坐标系说明：
    #   - 原始点云的 Y 轴正方向 = mesh 采样点的 Z 轴负方向
    #   - 原始点云的 Z 轴正方向 = mesh 的 Y 轴正方向
    #
    # 因此设原始点 (x_p, y_p, z_p)，对应到 mesh 坐标 (x_m, y_m, z_m) 为：
    #   x_m = x_p
    #   y_m = z_p
    #   z_m = -y_p
    # 即：
    #   [x_m, y_m, z_m] = [x_p, z_p, -y_p]
    points_aligned = np.empty_like(points)
    points_aligned[:, 0] = points[:, 0]          # X 保持不变
    points_aligned[:, 1] = points[:, 2]          # Y <- 原始 Z
    points_aligned[:, 2] = -points[:, 1]         # Z <- - 原始 Y

    # 使用点云自身做归一化（类似 renderer.normalize_model，但不直接用其矩阵）
    min_bound = points_aligned.min(axis=0)
    max_bound = points_aligned.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    max_extent = np.max(extent)

    if max_extent < 1e-6:
        print("  警告：点云范围过小，跳过归一化，仅做坐标轴对齐")
        points = points_aligned.astype(np.float32)
    else:
        scale = 1.0 / max_extent
        points = (points_aligned - center) * scale
        points = points.astype(np.float32)
        print("  对点云进行了坐标轴对齐并归一化")
    
    # 2. 渲染或加载图像
    if use_runtime_rendering:
        # 运行时渲染
        print("  使用运行时渲染...")
        glb_path = os.path.join(glb_dir, f"{object_id}.glb")
        if not os.path.exists(glb_path):
            print(f"错误：找不到GLB文件 {glb_path}")
            return
        
        # 初始化渲染器
        renderer = Open3DRenderer(width=image_size, height=image_size)
        renderer.setup()
        
        # 加载和归一化模型
        model = renderer.load_model(glb_path)
        model, norm_transform = renderer.normalize_model(model)

        # 在 GLB 归一化完成后，定量评估“点云 vs GLB”之间的差异（例如 fitness）
        try:
            mesh_for_eval = renderer.get_mesh_from_model(model)
            if mesh_for_eval is not None and len(mesh_for_eval.vertices) > 0:
                # 从 GLB 采样点作为 ICP/评价的目标点云
                mesh_sampled = mesh_for_eval.sample_points_uniformly(number_of_points=10000)

                # 用当前坐标系下的点云构造 Open3D PointCloud
                pcd_eval = o3d.geometry.PointCloud()
                pcd_eval.points = o3d.utility.Vector3dVector(points.astype(np.float64))

                # 只做评价，不跑 ICP：使用单位变换
                max_corr_dist = 0.05
                eval_res = o3d.pipelines.registration.evaluate_registration(
                    pcd_eval,
                    mesh_sampled,
                    max_corr_dist,
                    np.eye(4),
                )
                print(
                    "  点云与归一化 GLB 之间的评价（当前对齐状态）："
                    f"fitness={eval_res.fitness:.4f}, "
                    f"inlier_rmse={eval_res.inlier_rmse:.6f}"
                )
            else:
                print("  警告：无法从 GLB 模型中提取用于评价的 mesh")
        except Exception as e:
            print(f"  警告：评估点云与 GLB 差异时出错: {e}")

        # 生成视角点
        viewpoints = o3d_sample_view_points(radius, partition)
        
        # 渲染所有视角
        images = []
        depth_maps = []
        print(f"  渲染 {len(viewpoints)} 个视角...")
        for viewpoint in tqdm(viewpoints, desc="渲染中", leave=False):
            img, depth = renderer.render_with_depth(model, viewpoint, return_depth=True)
            images.append(img)
            depth_maps.append(depth)
        
        images = np.array(images)  # (V, H, W, 3)
        print(f"  渲染完成: {images.shape}")
        
        # 清理渲染器
        renderer.cleanup()
    else:
        # 加载预渲染图像
        render_path = os.path.join(render_dir, object_id)
        if not os.path.exists(render_path):
            print(f"警告：渲染目录不存在 {render_path}，跳过")
            return
        
        images = load_rendered_images(render_path)
        print(f"  加载图像: {images.shape}")
        
        # 检查并调整预渲染图像的尺寸
        original_h, original_w = images.shape[1], images.shape[2]
        if original_h != image_size or original_w != image_size:
            print(f"  预渲染图像尺寸 ({original_h}x{original_w}) 与目标尺寸 ({image_size}x{image_size}) 不匹配")
            print(f"  调整图像尺寸到 {image_size}x{image_size}...")
            # 使用PIL进行高质量resize
            resized_images = []
            for img in images:
                pil_img = Image.fromarray(img.astype(np.uint8))
                pil_img = pil_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                resized_images.append(np.array(pil_img))
            images = np.array(resized_images)
            print(f"  调整后图像尺寸: {images.shape}")
        
        depth_maps = None  # 预渲染图像没有深度图
        
        # 生成相机参数（如果不存在）
        camera_params_path = os.path.join(render_path, 'camera_params.json')
        if os.path.exists(camera_params_path):
            camera_params = load_camera_params(camera_params_path)
            viewpoints = np.array(camera_params['viewpoints'])
        else:
            # 从render_open3d_offscreen.py生成视角点
            from render_open3d_offscreen import sample_view_points
            viewpoints = sample_view_points(radius, partition)
            viewpoints = viewpoints[:len(images)]  # 只使用实际渲染的视角数
            save_camera_params(render_path, viewpoints, camera_params_path)
            camera_params = load_camera_params(camera_params_path)
    
    # 创建相机内参和外参列表
    if use_runtime_rendering:
        # 运行时渲染：直接创建相机参数
        camera_params = {
            'intrinsic': {
                'width': image_size,
                'height': image_size,
                'fx': image_size / (2.0 * np.tan(np.radians(60.0) / 2.0)),
                'fy': image_size / (2.0 * np.tan(np.radians(60.0) / 2.0)),
                'cx': image_size / 2.0,
                'cy': image_size / 2.0,
                'fov': 60.0
            }
        }
        camera_intrinsic = create_camera_intrinsic_from_params(camera_params)
    else:
        # 预渲染：从保存的参数加载
        camera_intrinsic = create_camera_intrinsic_from_params(camera_params)
    
    camera_intrinsics = [camera_intrinsic] * len(viewpoints)
    camera_extrinsics = [create_camera_extrinsic_from_viewpoint(vp) for vp in viewpoints]
    
    # 4. 初始化DINO模型（参考pca.ipynb）
    print("  初始化DINO模型...")
    # 默认使用DINOv3 ViT-L模型（与pca.ipynb一致）
    # dinov3_location = os.getenv("DINOV3_LOCATION", "facebookresearch/dinov3")
    dinov3_location = "/mnt/extra/my_task/checkpoint/dinov3-vit7b16-pretrain-lvd1689m"
    
    # 检测可用的GPU数量
    if device == "cuda" and torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        num_gpus_to_use = min(4, num_available_gpus)  # 最多使用4个GPU（cuda:0到cuda:3）
        print(f"  检测到 {num_available_gpus} 个GPU，将使用 {num_gpus_to_use} 个GPU进行并行处理")
        
        # 多GPU模式：为每个GPU创建模型实例
        models = []
        devices = []
        for gpu_idx in range(num_gpus_to_use):
            torch_device = torch.device(f"cuda:{gpu_idx}")
            print(f"    在 {torch_device} 上加载模型...")
            model = AutoModel.from_pretrained(dinov3_location)
            model = model.to(torch_device)
            model.eval()
            models.append(model)
            devices.append(torch_device)
        
        # 5. 提取DINO特征（多GPU并行处理）
        print("  提取DINO特征...")
        dino_features = extract_dino_features_from_images(
            images, 
            models=models, 
            devices=devices,
            num_gpus=num_gpus_to_use
        )
    else:
        # 单GPU模式
        if device == "cuda":
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device(device)
        
        print(f"  在 {torch_device} 上加载模型...")
        model = AutoModel.from_pretrained(dinov3_location)
        model = model.to(torch_device)
        model.eval()
        
        # 5. 提取DINO特征（单GPU模式）
        print("  提取DINO特征...")
        dino_features = extract_dino_features_from_images(images, model=model, device=torch_device)
    print(f"  DINO特征形状: {dino_features.shape}")
    
    # 计算实际的patch数量（用于验证）
    actual_num_patches = dino_features.shape[1]
    expected_num_patches = (image_size // patch_size) ** 2
    print(f"  实际patch数: {actual_num_patches}, 期望: {expected_num_patches}")
    
    # 5.5. 保存2D中间结果（渲染图、深度图、DINO特征PCA可视化）
    save_2d_intermediate_results(
        images,
        depth_maps,
        dino_features,
        output_dir,
        object_id,
        points,
        camera_intrinsics,
        camera_extrinsics,
        image_size=image_size,
        patch_size=patch_size
    )
    
    # 6. 反投影特征到点云
    print("  反投影特征到点云...")
    # 自动检测图像尺寸
    actual_image_size = images.shape[1]
    point_features, visibility_count = backproject_features_to_points(
        points,
        images,
        camera_intrinsics,
        camera_extrinsics,
        dino_features,
        image_size=actual_image_size,  # 使用实际图像尺寸
        patch_size=patch_size,
        only_visible=True,
        depth_maps=depth_maps,  # 传递深度图用于精确可见性检测
        depth_threshold=depth_threshold
    )
    print(f"  点特征形状: {point_features.shape}")
    print(f"  可见点数量: {(visibility_count > 0).sum()}/{len(points)}")
    
    # 7. PCA可视化
    print("  进行PCA可视化...")
    os.makedirs(output_dir, exist_ok=True)
    output_pcd_path = os.path.join(output_dir, f"{object_id}_pca.ply")
    pcd = visualize_features_with_pca(points, point_features, output_pcd_path)
    
    # 8. 保存特征（可选）
    feature_output_path = os.path.join(output_dir, f"{object_id}_features.npy")
    np.save(feature_output_path, point_features)
    print(f"  保存特征到: {feature_output_path}")
    
    print(f"✓ 完成: {object_id}")


def main():
    """主函数：批量处理所有对象"""
    import argparse
    
    parser = argparse.ArgumentParser(description='提取DINO特征并反投影到点云')
    parser.add_argument('--object_id', type=str, default=None,
                       help='处理单个对象ID（如果未指定，则处理所有对象）')
    parser.add_argument('--npy_dir', type=str, default='example_material/npys',
                       help='点云目录')
    parser.add_argument('--render_dir', type=str, default='example_material/renders_o3d',
                       help='渲染图像目录')
    parser.add_argument('--output_dir', type=str, default='example_material/dino_features',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--image_size', type=int, default=1120,
                       help='图像尺寸（会自动调整为patch_size的倍数，默认1120=16*70）')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='DINO patch尺寸（DINOv3默认16）')
    parser.add_argument('--use_runtime_rendering', action='store_true', default=True,
                       help='使用运行时渲染（默认True）')
    parser.add_argument('--no_runtime_rendering', dest='use_runtime_rendering', action='store_false',
                       help='不使用运行时渲染，使用预渲染图像')
    parser.add_argument('--depth_threshold', type=float, default=0.05,
                       help='深度阈值（用于可见性检测）')
    
    args = parser.parse_args()
    
    # 获取所有对象ID
    npy_dir = Path(args.npy_dir)
    if args.object_id:
        object_ids = [args.object_id]
    else:
        object_ids = [f.stem.replace('_8192', '') for f in npy_dir.glob('*.npy')]
    
    print(f"找到 {len(object_ids)} 个对象")
    
    # 处理每个对象
    for obj_id in tqdm(object_ids, desc="处理对象"):
        try:
            process_single_object(
                obj_id,
                npy_dir=args.npy_dir,
                glb_dir='example_material/glbs',  # 默认GLB目录
                render_dir=args.render_dir,
                output_dir=args.output_dir,
                device=args.device,
                image_size=args.image_size,
                patch_size=args.patch_size,
                use_runtime_rendering=args.use_runtime_rendering,
                depth_threshold=args.depth_threshold
            )
        except Exception as e:
            print(f"✗ 处理失败 {obj_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

