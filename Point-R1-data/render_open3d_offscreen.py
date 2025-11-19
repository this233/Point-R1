"""
使用Open3D的OffscreenRenderer进行渲染
参考back-to-3d仓库的渲染视角设定方式
复用renderer_o3d模块中的功能
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# 从renderer_o3d模块导入可复用的功能
from renderer_o3d import (
    Open3DRenderer,
    sample_view_points,
    create_camera_extrinsic_from_viewpoint
)


def render_model_from_viewpoint(model, viewpoint, output_path, renderer=None, width=1024, height=1024, save_depth=True):
    """
    从指定视角渲染模型（保留材质和纹理）
    
    使用Open3DRenderer进行渲染，复用renderer_o3d模块的功能
    
    参数:
        model: Open3D TriangleMeshModel对象
        viewpoint: 相机位置 (3,)
        output_path: 输出图像路径（RGB图像）
        renderer: Open3DRenderer实例（如果提供，则复用；否则创建新的）
        width: 图像宽度（像素）
        height: 图像高度（像素）
        save_depth: 是否保存深度图
    """
    # 如果没有提供renderer，创建新的
    if renderer is None:
        renderer = Open3DRenderer(width=width, height=height)
        renderer.setup()
        cleanup_renderer = True
    else:
        cleanup_renderer = False
    
    # 渲染图像和深度图
    img_array, depth_map = renderer.render_with_depth(model, viewpoint, center=None, return_depth=True)
    
    # 检查图像是否全白（可能表示渲染失败）
    if np.all(img_array[:, :, :3] >= 250):  # 检查RGB通道是否几乎全白
        print(f"警告: {output_path} 可能渲染失败（全白图像）")
        print(f"  相机位置: {viewpoint}")
    
    # 保存RGB图像到文件
    image = o3d.geometry.Image(img_array.astype(np.uint8))
    o3d.io.write_image(output_path, image)
    
    # 保存深度图（如果启用）
    if save_depth and depth_map is not None:
        # 生成深度图文件路径（将.png替换为_depth.npy或_depth.exr）
        depth_path = output_path.replace('.png', '_depth.npy')
        # 保存为numpy数组（保留原始深度值）
        np.save(depth_path, depth_map)
        
        # 同时保存为可视化图像（可选，用于调试）
        depth_vis_path = output_path.replace('.png', '_depth.png')
        # 归一化深度图到0-255范围用于可视化
        if depth_map.max() > depth_map.min():
            depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        depth_image = o3d.geometry.Image(depth_normalized)
        o3d.io.write_image(depth_vis_path, depth_image)
    
    # 只有在创建了新renderer时才清理
    if cleanup_renderer:
        renderer.cleanup()


def calculate_optimal_radius(bbox, fov_degrees=60.0, margin_factor=1.2):
    """
    根据模型的边界框和视场角计算最优相机距离
    
    确保模型完全在相机视野内，且有适当的边距
    
    参数:
        bbox: Open3D AxisAlignedBoundingBox对象
        fov_degrees: 垂直视场角（度）
        margin_factor: 边距系数，确保模型完全在视野内（默认1.2，即留20%边距）
    
    返回:
        float: 最优相机距离（从原点到相机的距离）
    """
    extent = bbox.get_extent()
    # 计算模型的最大尺寸（对角线长度）
    max_dimension = np.linalg.norm(extent)
    
    # 根据视场角计算相机距离
    # 几何关系：tan(fov/2) = (模型尺寸/2) / 相机距离
    # 推导：相机距离 = (模型尺寸/2) / tan(fov/2)
    fov_rad = np.radians(fov_degrees)
    optimal_distance = (max_dimension / 2.0) / np.tan(fov_rad / 2.0)
    
    # 应用边距系数，确保模型完全在视野内且不会被裁剪
    optimal_distance *= margin_factor
    
    return optimal_distance


def render_glb_file(glb_path, output_dir, radius=None, partition=5, num_views=None, auto_radius=True, save_depth=True):
    """
    渲染GLB文件的多视角图像
    
    完整的渲染流程：加载模型 -> 归一化 -> 生成视角 -> 逐个渲染
    复用renderer_o3d模块的功能
    
    参数:
        glb_path: GLB文件路径
        output_dir: 输出目录（将保存所有渲染图像和深度图）
        radius: 相机距离半径（如果为None且auto_radius=True，则自动计算）
        partition: 视角细分级别（控制视角数量）
        num_views: 视角数量（如果指定，则只使用前num_views个视角）
        auto_radius: 是否自动计算相机距离（基于模型尺寸）
        save_depth: 是否保存深度图（默认True）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用Open3DRenderer加载和归一化模型
    print(f"加载GLB文件: {glb_path}")
    renderer = Open3DRenderer()
    model = renderer.load_model(glb_path)
    model, norm_transform = renderer.normalize_model(model)
    
    # 检查模型是否有效（统计顶点和三角形数量）
    total_vertices = sum(len(mi.mesh.vertices) for mi in model.meshes)
    total_triangles = sum(len(mi.mesh.triangles) for mi in model.meshes)
    
    if total_vertices == 0:
        raise ValueError(f"Model没有顶点: {glb_path}")
    if total_triangles == 0:
        raise ValueError(f"Model没有三角形: {glb_path}")
    
    # 计算整体边界框（用于自动计算相机距离和打印信息）
    bbox = None
    for mi in model.meshes:
        mesh_bbox = mi.mesh.get_axis_aligned_bounding_box()
        if bbox is None:
            bbox = mesh_bbox
        else:
            min_bound = np.minimum(bbox.get_min_bound(), mesh_bbox.get_min_bound())
            max_bound = np.maximum(bbox.get_max_bound(), mesh_bbox.get_max_bound())
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # 打印模型信息（用于调试）
    print(f"  Mesh数量: {len(model.meshes)}, 材质数量: {len(model.materials)}")
    print(f"  总顶点数: {total_vertices}, 总三角形数: {total_triangles}")
    print(f"  Bounding box中心: {bbox.get_center()}, 尺寸: {bbox.get_extent()}")
    
    # 自动计算或使用指定的相机距离
    if auto_radius and radius is None:
        # 根据模型尺寸自动计算最优距离
        radius = calculate_optimal_radius(bbox)
        print(f"  自动计算相机距离: {radius:.3f}")
    elif radius is None:
        # 使用默认值
        radius = 1.1
        print(f"  使用默认相机距离: {radius:.3f}")
    else:
        # 使用用户指定的距离
        print(f"  使用指定相机距离: {radius:.3f}")
    
    # 生成视角点（复用renderer_o3d的函数）
    viewpoints = sample_view_points(radius, partition)
    if num_views is not None:
        # 如果指定了视角数量，只使用前N个
        viewpoints = viewpoints[:num_views]
    
    print(f"生成 {len(viewpoints)} 个视角，半径: {radius:.3f}")
    
    # 设置渲染器（复用同一个实例以提高效率）
    renderer.setup()
    
    # 为每个视角渲染图像（复用同一个renderer实例）
    for i, viewpoint in enumerate(tqdm(viewpoints, desc="渲染中")):
        output_path = os.path.join(output_dir, f"view_{i:03d}.png")
        render_model_from_viewpoint(model, viewpoint, output_path, renderer=renderer, save_depth=save_depth)
    
    # 清理渲染器
    renderer.cleanup()
    
    print(f"渲染完成，输出目录: {output_dir}")


def main():
    """
    主函数：批量渲染glbs目录下的所有GLB文件
    
    支持命令行参数配置渲染选项
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用Open3D OffscreenRenderer渲染GLB文件')
    parser.add_argument('--glb_dir', type=str, default='./example_material/glbs',
                       help='GLB文件目录')
    parser.add_argument('--output_dir', type=str, default='./example_material/renders_o3d',
                       help='输出目录')
    parser.add_argument('--radius', type=float, default=None,
                       help='相机距离半径（如果未指定，将自动计算）')
    parser.add_argument('--partition', type=int, default=5,
                       help='视角细分级别（控制视角密度）')
    parser.add_argument('--num_views', type=int, default=None,
                       help='视角数量（如果指定，则只使用前N个视角）')
    parser.add_argument('--no_auto_radius', action='store_true',
                       help='禁用自动计算相机距离（使用默认值1.1）')
    parser.add_argument('--no_depth', dest='save_depth', action='store_false', default=True,
                       help='不保存深度图（默认保存）')
    
    args = parser.parse_args()
    
    # 获取所有GLB文件
    glb_dir = Path(args.glb_dir)
    glb_files = list(glb_dir.glob("*.glb"))
    
    if len(glb_files) == 0:
        print(f"在 {glb_dir} 中没有找到GLB文件")
        return
    
    print(f"找到 {len(glb_files)} 个GLB文件")
    
    # 为每个GLB文件创建输出目录并渲染
    for glb_file in glb_files:
        model_name = glb_file.stem  # 不含扩展名的文件名
        output_dir = os.path.join(args.output_dir, model_name)
        
        try:
            # 渲染单个GLB文件
            render_glb_file(
                str(glb_file),
                output_dir,
                radius=args.radius,
                partition=args.partition,
                num_views=args.num_views,
                auto_radius=not args.no_auto_radius,
                save_depth=args.save_depth
            )
            print(f"✓ 完成: {glb_file.name}")
        except Exception as e:
            # 捕获并打印错误，继续处理下一个文件
            print(f"✗ 失败: {glb_file.name}, 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
