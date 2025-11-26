import os
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import cv2
import clustering_utils
from renderer_o3d import (
    Open3DRenderer,
    check_visible_points_with_depth,
    project_points_to_image_with_depth,
    sample_view_points,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint
)

def load_pointcloud(npy_path):
    """加载点云数据"""
    points = np.load(npy_path)
    if points.shape[1] >= 3:
        return points[:, :3].astype(np.float32)
    else:
        return points.astype(np.float32)

def preprocess_pointcloud(points):
    """
    对点云做坐标轴对齐 + 自身归一化，使其与 renderer 中 normalized GLB 坐标系一致
    参考 extract_dino_features.py 中的实现
    """
    # 1. 坐标轴对齐
    points_aligned = np.empty_like(points)
    points_aligned[:, 0] = points[:, 0]          # X 保持不变
    points_aligned[:, 1] = points[:, 2]          # Y <- 原始 Z
    points_aligned[:, 2] = -points[:, 1]         # Z <- - 原始 Y

    # 2. 自身归一化
    min_bound = points_aligned.min(axis=0)
    max_bound = points_aligned.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    max_extent = np.max(extent)

    if max_extent < 1e-6:
        print("警告：点云范围过小，跳过归一化")
        return points_aligned.astype(np.float32)
    else:
        scale = 1.0 / max_extent
        points_norm = (points_aligned - center) * scale
        return points_norm.astype(np.float32)

def draw_cluster_contour(image, points_2d, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制点云簇的轮廓
    """
    H, W = image.shape[:2]
    
    # 创建掩码
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 在掩码上绘制点（膨胀以连接点）
    for x, y in points_2d:
        cv2.circle(mask, (int(x), int(y)), radius=5, color=255, thickness=-1)
    
    # 闭运算连接断开的区域
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制轮廓
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='可视化单个聚类的轮廓描边')
    parser.add_argument('--glb_path', type=str, default='example_material/glbs/d1e62f1eb6334a1196b53d61529f472b.glb', help='GLB文件路径')
    parser.add_argument('--npy_path', type=str, default='example_material/npys/d1e62f1eb6334a1196b53d61529f472b_8192.npy', help='点云文件路径')
    parser.add_argument('--feat_path', type=str, default='example_material/dino_features/d1e62f1eb6334a1196b53d61529f472b_features.npy', help='特征文件路径')
    parser.add_argument('--output_dir', type=str, default='output_visualization', help='输出目录')
    parser.add_argument('--cluster_id', type=int, default=-1, help='要可视化的簇ID，-1表示可视化最大的簇')
    parser.add_argument('--alpha', type=float, default=0.04, help='聚类参数 Alpha')
    parser.add_argument('--image_size', type=int, default=1024, help='渲染图像尺寸')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载数据
    print(f"加载 GLB: {args.glb_path}")
    print(f"加载 PC: {args.npy_path}")
    print(f"加载 Feat: {args.feat_path}")
    
    points_raw = load_pointcloud(args.npy_path)
    features = np.load(args.feat_path)
    
    # 2. 预处理点云 (对齐 + 归一化)
    points = preprocess_pointcloud(points_raw)
    print(f"点云已预处理: {points.shape}")
    
    # 3. 执行聚类
    # Alpha -> K
    k_neighbors = int(10 + args.alpha * 500)
    betas = [0.0, 0.3, 0.5, 0.7] # Level 1-4
    print(f"执行聚类 (K={k_neighbors})...")
    clustering_results = clustering_utils.perform_hierarchical_clustering(
        points, features, k_neighbors, betas
    )
    
    # 使用 Level 1 (beta=0.3) 的结果
    labels = clustering_results[1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"聚类完成，共 {len(unique_labels)} 个簇")
    
    # 选择簇
    if args.cluster_id == -1:
        # 选择最大的簇
        target_cluster_id = unique_labels[np.argmax(counts)]
        print(f"自动选择最大的簇 ID: {target_cluster_id} (包含 {np.max(counts)} 个点)")
    else:
        target_cluster_id = args.cluster_id
        if target_cluster_id not in unique_labels:
            print(f"错误: 簇 ID {target_cluster_id} 不存在")
            return
    
    # 获取目标簇的点索引
    cluster_indices = np.where(labels == target_cluster_id)[0]
    cluster_points = points[cluster_indices]
    print(f"目标簇包含 {len(cluster_points)} 个点")
    
    # 4. 初始化渲染器
    renderer = Open3DRenderer(width=args.image_size, height=args.image_size)
    renderer.setup()
    
    model = renderer.load_model(args.glb_path)
    model, _ = renderer.normalize_model(model)
    
    # 5. 生成视角
    radius = 1.5 # 相机距离
    partition = 4
    viewpoints = sample_view_points(radius, partition)
    
    print(f"开始渲染和投影，共 {len(viewpoints)} 个视角...")
    
    for i, viewpoint in enumerate(tqdm(viewpoints)):
        # 渲染
        img, depth_map = renderer.render_with_depth(model, viewpoint, return_depth=True)
        
        # 投影
        camera_params = {
            'intrinsic': {
                'width': args.image_size,
                'height': args.image_size,
                'fx': args.image_size / (2.0 * np.tan(np.radians(60.0) / 2.0)),
                'fy': args.image_size / (2.0 * np.tan(np.radians(60.0) / 2.0)),
                'cx': args.image_size / 2.0,
                'cy': args.image_size / 2.0,
                'fov': 60.0
            }
        }
        intrinsic = create_camera_intrinsic_from_params(camera_params)
        extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint)
        
        # 投影所有点以检查可见性
        pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
            cluster_points, intrinsic, extrinsic, image_size=args.image_size
        )
        
        # 检查可见性
        # 1. 投影在图像范围内
        # 2. 深度一致性
        
        # 使用 check_visible_points_with_depth
        # 注意: check_visible_points_with_depth 需要完整的点云输入逻辑，这里我们只传了 cluster_points
        # 只要 pixel_coords 和 depths 是对应的即可。
        
        is_visible = check_visible_points_with_depth(
            cluster_points, 
            pixel_coords, 
            depths, 
            depth_map, 
            use_relative_threshold=True, 
            relative_threshold_ratio=0.05
        )
        
        # 最终可见点
        visible_indices = np.where(valid_mask & is_visible)[0]
        
        # 如果可见点太少，跳过或标记
        if len(visible_indices) < 10:
            continue
            
        visible_pixels = pixel_coords[visible_indices]
        
        # 绘制轮廓
        # img 是 RGB (H, W, 3)
        result_img = draw_cluster_contour(img, visible_pixels, color=(0, 255, 0), thickness=3)
        
        # 保存
        output_filename = os.path.join(args.output_dir, f"cluster_{target_cluster_id}_view_{i:03d}.png")
        cv2.imwrite(output_filename, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
    renderer.cleanup()
    print(f"完成。结果保存在 {args.output_dir}")

if __name__ == "__main__":
    main()







