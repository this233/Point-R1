"""
Open3D运行时渲染模块
支持实时渲染并返回图像和深度信息，用于特征提取和可见性检测
"""

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from typing import Tuple, Optional, List
from scipy.spatial.transform import Rotation as R
# from extract_dino_features import create_camera_extrinsic_from_viewpoint

    

class Open3DRenderer:
    """Open3D渲染器，支持运行时渲染和深度图"""
    
    def __init__(self, width=1024, height=1024, device='cpu'):
        """
        初始化渲染器
        
        参数:
            width: 图像宽度
            height: 图像高度
            device: 设备（Open3D主要在CPU上运行）
        """
        self.width = width
        self.height = height
        self.device = device
        self.renderer = None
        self.scene = None
        
    def setup(self):
        """设置渲染器和场景"""
        self.renderer = rendering.OffscreenRenderer(self.width, self.height)
        self.scene = self.renderer.scene
        self.scene.set_background([1.0, 1.0, 1.0, 1.0])  # 白色背景

    def load_model(self, glb_path):
        """
        加载GLB模型
        """
        model = o3d.io.read_triangle_model(glb_path)
        if model is None or len(model.meshes) == 0:
            raise ValueError(f"无法加载GLB文件或文件中没有mesh: {glb_path}")
        return model
        
    def upload_model(self, model):
        """
        将模型加载到场景中（仅需执行一次）
        """
        if self.renderer is None:
            self.setup()
            
        # 清空场景
        self.scene.clear_geometry()
        
        # 添加mesh到场景
        for mi in model.meshes:
            mesh = mi.mesh
            mat = model.materials[mi.material_idx]
            self.scene.add_geometry(mi.mesh_name, mesh, mat)

    def render_view(self, viewpoint, center=None, return_depth=True):
        """
        仅设置相机并渲染（假设模型已加载）
        """
        if self.renderer is None:
            self.setup()

        # 计算观察中心
        # 注意：这里不再自动计算 bbox center，因为 model 不在参数里
        # 如果 center 为 None，默认为原点，或者调用者必须提供
        if center is None:
             center = np.array([0, 0, 0])
        
        # 设置相机：显式使用针孔相机内参 + 外参
        fov = 60.0
        cam_params = {
            "intrinsic": {
                "width": self.width,
                "height": self.height,
                "fx": self.width / (2.0 * np.tan(np.radians(fov) / 2.0)),
                "fy": self.height / (2.0 * np.tan(np.radians(fov) / 2.0)),
                "cx": self.width / 2.0,
                "cy": self.height / 2.0,
                "fov": fov,
            }
        }
        intrinsic = create_camera_intrinsic_from_params(cam_params)
        extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
        
        self.renderer.setup_camera(intrinsic, extrinsic)
        
        image = self.renderer.render_to_image()
        img_array = np.asarray(image)[:, :, :3]  # (H, W, 3)
        
        depth = None
        if return_depth:
            depth_image = self.renderer.render_to_depth_image(z_in_view_space=True)
            depth = np.asarray(depth_image)
            depth = np.abs(depth)
            depth[~np.isfinite(depth)] = 0
            depth[depth > 1000] = 0
        
        return img_array, depth
    
    def render_with_depth(self, model, viewpoint, center=None, return_depth=True):
        """
        (旧接口兼容) 从指定视角渲染模型，返回图像和深度图
        """
        self.upload_model(model)
        
        # 如果 center 为空，计算模型中心
        if center is None:
            bbox = None
            for mi in model.meshes:
                mesh_bbox = mi.mesh.get_axis_aligned_bounding_box()
                if bbox is None:
                    bbox = mesh_bbox
                else:
                    min_bound = np.minimum(bbox.get_min_bound(), mesh_bbox.get_min_bound())
                    max_bound = np.maximum(bbox.get_max_bound(), mesh_bbox.get_max_bound())
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            center = bbox.get_center()
            
        return self.render_view(viewpoint, center, return_depth)
        """
        加载GLB模型
        
        参数:
            glb_path: GLB文件路径
        
        返回:
            model: Open3D TriangleMeshModel对象
        """
        model = o3d.io.read_triangle_model(glb_path)
        if model is None or len(model.meshes) == 0:
            raise ValueError(f"无法加载GLB文件或文件中没有mesh: {glb_path}")
        return model
    
    def normalize_model(self, model):
        """
        归一化模型到单位立方体
        
        参数:
            model: Open3D TriangleMeshModel对象
        
        返回:
            normalized_model: 归一化后的模型
            transform: 归一化变换矩阵
        """
        # 计算整体边界框
        bbox = None
        for mi in model.meshes:
            mesh_bbox = mi.mesh.get_axis_aligned_bounding_box()
            if bbox is None:
                bbox = mesh_bbox
            else:
                min_bound = np.minimum(bbox.get_min_bound(), mesh_bbox.get_min_bound())
                max_bound = np.maximum(bbox.get_max_bound(), mesh_bbox.get_max_bound())
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        
        if max_extent < 1e-6:
            return model, np.eye(4)
        
        scale = 1.0 / max_extent
        
        # 应用归一化
        for mi in model.meshes:
            mi.mesh.translate(-center)
            mi.mesh.scale(scale, center=(0, 0, 0))
            if not mi.mesh.has_vertex_normals():
                mi.mesh.compute_vertex_normals()
        
        transform = np.eye(4)
        transform[:3, 3] = -center
        transform = np.diag([scale, scale, scale, 1.0]) @ transform
        
        return model, transform
    
    def render_with_depth(self, model, viewpoint, center=None, return_depth=True):
        """
        从指定视角渲染模型，返回图像和深度图
        
        参数:
            model: Open3D TriangleMeshModel对象
            viewpoint: 相机位置 (3,)
            center: 观察中心（如果为None，则使用模型中心）
            return_depth: 是否返回深度图
        
        返回:
            image: RGB图像 (H, W, 3) numpy数组，值域[0, 255]
            depth: 深度图 (H, W) numpy数组（如果return_depth=True）
        """
        if self.renderer is None:
            self.setup()
        
        # 清空场景
        self.scene.clear_geometry()
        
        # 添加mesh到场景
        for mi in model.meshes:
            mesh = mi.mesh
            mat = model.materials[mi.material_idx]
            self.scene.add_geometry(mi.mesh_name, mesh, mat)
        
        # 计算观察中心
        if center is None:
            bbox = None
            for mi in model.meshes:
                mesh_bbox = mi.mesh.get_axis_aligned_bounding_box()
                if bbox is None:
                    bbox = mesh_bbox
                else:
                    min_bound = np.minimum(bbox.get_min_bound(), mesh_bbox.get_min_bound())
                    max_bound = np.maximum(bbox.get_max_bound(), mesh_bbox.get_max_bound())
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            center = bbox.get_center()
        
        # 设置相机：显式使用针孔相机内参 + 外参，保证与点云投影时使用的相机参数一致
        fov = 60.0
        cam_params = {
            "intrinsic": {
                "width": self.width,
                "height": self.height,
                "fx": self.width / (2.0 * np.tan(np.radians(fov) / 2.0)),
                "fy": self.height / (2.0 * np.tan(np.radians(fov) / 2.0)),
                "cx": self.width / 2.0,
                "cy": self.height / 2.0,
                "fov": fov,
            }
        }
        intrinsic = create_camera_intrinsic_from_params(cam_params)
        extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
        # 这里的 extrinsic 定义与 project_points_to_image_with_depth 中使用的一致
        self.renderer.setup_camera(intrinsic, extrinsic)
        # self.renderer.setup_camera(fov, at, eye, up) # 原先版本，用于测试
        image = self.renderer.render_to_image()
        img_array = np.asarray(image)[:, :, :3]  # (H, W, 3)
        
        depth = None
        if return_depth:
            # 渲染深度图
            # Open3D 文档：
            #   - 默认情况下，像素在 [0, 1]（near→far）的归一化深度
            #   - 当 z_in_view_space=True 时，像素预先变换到 view space，
            #     即相机坐标系下的 z 分量（距离相机沿前向轴的深度）
            # 为了和我们自己计算的点深度一致，这里使用 view-space z。
            depth_image = self.renderer.render_to_depth_image(z_in_view_space=True)
            depth = np.asarray(depth_image)  # (H, W)
            # 一般情况下，物体在相机前方，view-space z 为正；
            # 如果存在数值异常（inf、nan 或极大值），视为背景，置 0。
            depth = np.abs(depth)
            depth[~np.isfinite(depth)] = 0
            depth[depth > 1000] = 0
        
        return img_array, depth
    
    def get_mesh_from_model(self, model):
        """
        从模型中提取合并的mesh（用于可见性检测）
        
        参数:
            model: Open3D TriangleMeshModel对象
        
        返回:
            mesh: 合并后的Open3D TriangleMesh对象
        """
        if len(model.meshes) == 0:
            return None
        
        # 合并所有mesh
        combined_mesh = model.meshes[0].mesh
        for i in range(1, len(model.meshes)):
            combined_mesh += model.meshes[i].mesh
        
        return combined_mesh
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.scene is not None:
                self.scene.clear_geometry()
        except Exception as e:
            print(f"Warning: Error during cleanup (ignored): {e}")
            
        self.renderer = None
        self.scene = None


def check_visible_points_with_depth(
    points: np.ndarray,
    pixel_coords: np.ndarray,
    depths: np.ndarray,
    depth_map: np.ndarray,
    depth_threshold: float = 0.01,
    use_relative_threshold: bool = True,
    relative_threshold_ratio: float = 0.05
) -> np.ndarray:
    """
    使用深度图检查点的可见性（参考back-to-3d的思路）
    
    参数:
        points: (N, 3) 3D点坐标
        pixel_coords: (N, 2) 像素坐标 [x, y]
        depths: (N,) 点到相机光心的欧氏距离（单位：米）
        depth_map: (H, W) 深度图
        depth_threshold: 深度阈值（绝对阈值，如果use_relative_threshold=False）
        use_relative_threshold: 是否使用相对阈值（相对于深度值的百分比）
        relative_threshold_ratio: 相对阈值比例（例如0.05表示5%）
    
    返回:
        visible_mask: (N,) 布尔数组，表示点是否可见
    """
    H, W = depth_map.shape
    visible_mask = np.zeros(len(points), dtype=bool)
    
    # 检查每个点
    for i in range(len(points)):
        x, y = pixel_coords[i]
        depth = depths[i]
        
        # 检查是否在图像范围内
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        
        # 获取深度图中的深度值
        x_int = int(np.clip(x, 0, W - 1))
        y_int = int(np.clip(y, 0, H - 1))
        depth_at_pixel = depth_map[y_int, x_int]
        
        # 如果深度图值为0或负值，表示该像素没有物体（背景）
        if depth_at_pixel <= 0:
            continue
        
        # 计算深度差异
        depth_diff = abs(depth - depth_at_pixel)
        
        # 使用相对阈值或绝对阈值
        if use_relative_threshold:
            # 使用相对阈值：深度差异相对于深度值的百分比
            # 取两个深度值的平均值作为参考
            avg_depth = (depth + depth_at_pixel) / 2.0
            threshold = avg_depth * relative_threshold_ratio
        else:
            # 使用绝对阈值
            threshold = depth_threshold
        
        # 如果点的深度接近深度图中的深度（在阈值内），则认为可见
        if depth_diff < threshold:
            visible_mask[i] = True
        # 如果点的深度明显大于深度图中的深度（点被遮挡），则不可见
        elif depth > depth_at_pixel + threshold:
            visible_mask[i] = False
    
    return visible_mask


def check_visible_points_with_mesh(
    points: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    pixel_coords: np.ndarray,
    visible_faces: np.ndarray,
    face_radius: float = 0.01
) -> np.ndarray:
    """
    使用mesh和可见面检查点的可见性（参考back-to-3d的实现）
    
    参数:
        points: (N, 3) 3D点坐标
        mesh: Open3D TriangleMesh对象
        pixel_coords: (N, 2) 像素坐标
        visible_faces: 可见面的索引数组
        face_radius: 判断点是否属于面的半径阈值
    
    返回:
        visible_mask: (N,) 布尔数组，表示点是否可见
    """
    if mesh is None or len(visible_faces) == 0:
        return np.zeros(len(points), dtype=bool)
    
    visible_mask = np.zeros(len(points), dtype=bool)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # 获取可见面的顶点
    visible_triangles = triangles[visible_faces]
    visible_vertices = np.unique(visible_triangles.flatten())
    
    # 对于每个点，检查是否接近可见面的顶点
    for i in range(len(points)):
        point = points[i]
        
        # 计算点到所有可见顶点的距离
        distances = np.linalg.norm(vertices[visible_vertices] - point, axis=1)
        min_dist = np.min(distances)
        
        # 如果最小距离小于阈值，则认为点可见
        if min_dist < face_radius:
            visible_mask[i] = True
    
    return visible_mask


def sample_view_points(radius, partition):
    """
    生成视角点（参考back-to-3d仓库的实现）
    
    参数:
        radius: 相机距离原点的半径
        partition: 视角细分级别，控制采样密度
    
    返回:
        np.array: 视角点数组 (N, 3)，每个点表示一个相机位置
    """
    points = []
    
    # 生成球面坐标的phi（方位角）和theta（极角）
    phi = np.linspace(0, 2 * np.pi, (partition + 1) * 2, endpoint=False)
    theta = np.linspace(0, np.pi, (partition + 1), endpoint=False)
    
    for i, p in enumerate(phi):
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.cos(t)
            z = radius * np.sin(t) * np.sin(p)
            if t == 0:
                continue
            points.append([x, y, z])
    
    points.append([0, radius, 0])
    points.append([0, -radius, 0])
    
    # 对所有点进行微小旋转
    rotation_matrix = R.from_rotvec(0.001 * np.array([1, 0, 0])).as_matrix()
    rotated_points = []
    for point in points:
        rotated_point = np.dot(rotation_matrix, point)
        rotated_points.append(rotated_point)
    
    return np.array(rotated_points)


def create_camera_intrinsic_from_params(params):
    """从参数创建Open3D相机内参对象"""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        params['intrinsic']['width'],
        params['intrinsic']['height'],
        params['intrinsic']['fx'],
        params['intrinsic']['fy'],
        params['intrinsic']['cx'],
        params['intrinsic']['cy']
    )
    return intrinsic


def create_camera_extrinsic_from_viewpoint(viewpoint, center=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """
    从视角点创建相机外参矩阵
    
    注意：此处的定义需要与特征反投影 / 点云投影时保持一致，
    因此与 extract_dino_features.py 中的实现对齐。
    """
    eye = viewpoint.astype(np.float64)
    at = center.astype(np.float64)
    up_vec = up.astype(np.float64)
    
    # 计算前向量
    forward = at - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0, 0, -1], dtype=np.float64)
    else:
        forward = forward / forward_norm
    
    # 计算右向量
    right = np.cross(forward, up_vec)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up_vec = np.array([0, 0, 1], dtype=np.float64)
        right = np.cross(forward, up_vec)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    
    # 重新计算上向量
    up_vec = np.cross(forward, right)
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    # 构建视图矩阵（与 extract_dino_features 中保持一致）
    R_cam = np.array([right, up_vec, forward])
    t_cam = -R_cam @ eye
    
    # 构建外参矩阵
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R_cam
    extrinsic[:3, 3] = t_cam
    
    return extrinsic


def project_points_to_image_with_depth(
    points: np.ndarray,
    camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    camera_extrinsic: np.ndarray,
    image_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将3D点投影到2D图像坐标，并计算深度
    
    参数:
        points: (N, 3) 3D点坐标
        camera_intrinsic: Open3D相机内参对象
        camera_extrinsic: (4, 4) 相机外参矩阵（世界到相机）
        image_size: 图像尺寸
    
    返回:
        pixel_coords: (N, 2) 像素坐标 [x, y]
        depths: (N,) 点深度（相机坐标系下的 z 分量，取绝对值，单位与场景一致）
        valid_mask: (N,) 布尔数组，表示点是否在图像范围内且深度为正（在相机前方）
    """
    # 转换为齐次坐标
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 转换到相机坐标系
    points_cam = (camera_extrinsic @ points_homo.T).T[:, :3]
    
    # 检查深度（z坐标）是否为正（点在相机前方）
    valid_depth = points_cam[:, 2] > 0
    
    # 投影到图像平面
    fx = camera_intrinsic.intrinsic_matrix[0, 0]
    fy = camera_intrinsic.intrinsic_matrix[1, 1]
    cx = camera_intrinsic.intrinsic_matrix[0, 2]
    cy = camera_intrinsic.intrinsic_matrix[1, 2]
    
    x = points_cam[:, 0] / points_cam[:, 2] * fx + cx
    y = points_cam[:, 1] / points_cam[:, 2] * fy + cy
    
    pixel_coords = np.stack([x, y], axis=1)
    # 为了和 render_to_depth_image(z_in_view_space=True) 的语义一致，
    # 这里将点深度定义为相机坐标系下的 z 分量（取绝对值）。
    depths = np.abs(points_cam[:, 2])
    
    # 检查点是否在图像范围内
    valid_range = (x >= 0) & (x < image_size) & (y >= 0) & (y < image_size)
    valid_mask = valid_depth & valid_range
    
    return pixel_coords, depths, valid_mask

