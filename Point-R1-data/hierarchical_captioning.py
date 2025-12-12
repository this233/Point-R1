"""
层级化 3D 物体标注脚本

功能：
1. 读取 .glb 和 .npy 文件，提取 DINO 特征
2. 执行层级聚类
3. 从第0级（全局）开始遍历层级树
4. 生成多视角渲染和 SoM 可视化
5. 调用 MLLM API 进行层级化标注

使用方法：
    python hierarchical_captioning.py \
        --glb_path example_material/glbs/xxx.glb \
        --npy_path example_material/npys/xxx_8192.npy \
        --output_dir outputs/captions
"""

import os
import sys
import json
import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import time

import numpy as np
import random
from PIL import Image
import cv2

# 复用现有模块
import clustering_utils
from renderer_o3d import (
    Open3DRenderer,
    check_visible_points_with_depth,
    project_points_to_image_with_depth,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint,
    sample_view_points
)

# 尝试导入 open3d
try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("警告: open3d 未安装")


# ===================== 数据结构 =====================

@dataclass
class ClusterCaption:
    """单个聚类簇的标注结果"""
    cluster_id: int
    som_id: int  # SoM 风格的编号 (1, 2, 3...)
    level: int
    parent_id: Optional[int]
    name: str  # 简短名称，如 "头部"
    caption: str  # 详细描述
    color: str  # 对应的颜色（用于 debug）
    point_count: int
    visible_ratio: float  # 在最佳视角下的可见比例
    children: List['ClusterCaption'] = field(default_factory=list)


@dataclass
class ClusterAnnotation:
    """单个视角下的簇标注"""
    cluster_id: int
    som_id: int
    name: str
    description: str
    color: str = ""


@dataclass
class ViewAnnotationResult:
    """单个视角的标注结果"""
    view_idx: int
    view_direction: str  # 视角方向描述（如：正面平视）
    annotations: List[ClusterAnnotation]



@dataclass
class HierarchicalCaptionResult:
    """完整的层级标注结果"""
    object_id: str
    global_name: str
    global_caption: str
    root_cluster: ClusterCaption
    total_clusters: int
    total_levels: int
    processing_time: float


# ===================== MLLM API 接口 =====================

class MLLMClient:
    """多模态大语言模型客户端基类"""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
    def encode_image(self, image: np.ndarray) -> str:
        """将图像编码为 base64"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def call(self, messages: List[Dict], max_tokens: int = 4096, **kwargs) -> str:
        """调用 API（子类实现）"""
        raise NotImplementedError
    

class OpenAIClient(MLLMClient):
    """OpenAI API 客户端（兼容格式）"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def call(self, messages: List[Dict], max_tokens: int = 4096, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content


class DashScopeClient(MLLMClient):
    """阿里云 DashScope API 客户端（Qwen3-VL 等模型）"""
    
    def __init__(self, api_key: str, model: str = "qwen3-vl-plus", 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 enable_thinking: bool = False, thinking_budget: int = 4096):
        super().__init__(api_key, model, base_url)
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def call(self, messages: List[Dict], max_tokens: int = 4096, **kwargs) -> str:
        """调用 DashScope API，支持流式响应和思考模式"""
        # 确定是否启用思考模式 (优先使用调用时的参数，否则使用实例默认值)
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)
        
        # 获取 temperature 参数（用于控制输出多样性）
        temperature = kwargs.get('temperature', 0.7)
        
        # 构建请求参数
        kwargs_api = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # 使用流式以支持 thinking
            "temperature": temperature,
        }
        
        # 如果启用思考模式
        if enable_thinking:
            kwargs_api["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": self.thinking_budget
            }
        
        # 发起流式请求
        completion = self.client.chat.completions.create(**kwargs_api)
        
        reasoning_content = ""
        answer_content = ""
        
        for chunk in completion:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            # 收集思考过程（如果有）
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            
            # 收集回答内容
            if delta.content is not None:
                answer_content += delta.content
        
        # 如果需要，可以记录思考过程
        if enable_thinking and reasoning_content:
            print(f"[Thinking] {reasoning_content}", flush=True)
        
        return answer_content


class AnthropicClient(MLLMClient):
    """Anthropic API 客户端"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", base_url: Optional[str] = None):
        super().__init__(api_key, model, base_url)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")
    
    def call(self, messages: List[Dict], max_tokens: int = 4096, **kwargs) -> str:
        # 转换消息格式
        system_msg = None
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                converted_messages.append(msg)
        
        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text


def create_mllm_client(provider: str, api_key: str, model: Optional[str] = None, 
                       base_url: Optional[str] = None,
                       enable_thinking: bool = False,
                       thinking_budget: int = 4096) -> MLLMClient:
    """创建 MLLM 客户端"""
    if provider == "openai":
        return OpenAIClient(api_key, model or "gpt-4o", base_url)
    elif provider == "anthropic":
        return AnthropicClient(api_key, model or "claude-sonnet-4-20250514", base_url)
    elif provider == "openai-compatible":
        return OpenAIClient(api_key, model or "gpt-4o", base_url)
    elif provider == "dashscope":
        return DashScopeClient(
            api_key, 
            model or "qwen3-vl-plus",
            base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget
        )
    else:
        raise ValueError(f"不支持的 provider: {provider}")


# ===================== Prompt 模板加载 =====================

# Prompt 文件目录
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(prompt_name: str) -> str:
    """从文件加载 prompt 模板"""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")


# 延迟加载 prompts（首次使用时加载）
_PROMPT_CACHE = {}

def get_prompt(prompt_name: str) -> str:
    """获取 prompt（带缓存）"""
    if prompt_name not in _PROMPT_CACHE:
        _PROMPT_CACHE[prompt_name] = load_prompt(prompt_name)
    return _PROMPT_CACHE[prompt_name]


# ===================== 核心功能函数 =====================

def convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def load_and_prepare_data(
    glb_path: str,
    npy_path: str,
    feature_path: Optional[str] = None,
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.3, 0.5, 0.7]
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Any]:
    """
    加载数据并执行聚类
    
    返回:
        points: 归一化后的点云坐标
        clustering_results: 各层级的聚类标签
        model: 归一化后的 GLB 模型
    """
    print(f"加载点云: {npy_path}")
    points_raw = np.load(npy_path)
    if points_raw.shape[1] >= 3:
        points_raw = points_raw[:, :3]
    
    # 坐标轴对齐和归一化（与 visualize_pointcloud_gradio.py 一致）
    points_aligned = np.empty_like(points_raw)
    points_aligned[:, 0] = points_raw[:, 0]
    points_aligned[:, 1] = points_raw[:, 2]
    points_aligned[:, 2] = -points_raw[:, 1]
    
    min_bound = points_aligned.min(axis=0)
    max_bound = points_aligned.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    max_extent = np.max(extent)
    
    if max_extent < 1e-6:
        points = points_aligned.astype(np.float32)
    else:
        scale = 1.0 / max_extent
        points = ((points_aligned - center) * scale).astype(np.float32)
    
    print(f"点云形状: {points.shape}")
    
    # 加载或提取特征
    if feature_path and os.path.exists(feature_path):
        print(f"加载特征: {feature_path}")
        features = np.load(feature_path)
    else:
        print("特征文件不存在，需要先提取特征")
        print(f"请运行: python extract_dino_features.py --object_id {Path(npy_path).stem.replace('_8192', '')}")
        raise FileNotFoundError(f"特征文件不存在: {feature_path}")
    
    # 执行聚类
    print(f"执行层级聚类 (K={k_neighbors}, Betas={betas})...")
    clustering_results = clustering_utils.perform_hierarchical_clustering(
        points, features, k_neighbors, betas
    )
    
    # 加载 GLB 模型
    print(f"加载 GLB 模型: {glb_path}")
    renderer = Open3DRenderer(width=800, height=800)
    renderer.setup()
    model = renderer.load_model(glb_path)
    model, _ = renderer.normalize_model(model)
    renderer.cleanup()
    
    return points, clustering_results, model


def get_viewpoint_from_angles(azimuth: float, elevation: float, radius: float) -> np.ndarray:
    """根据方位角、仰角和半径计算相机位置"""
    theta = np.radians(90 - elevation)
    phi = np.radians(azimuth)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    
    return np.array([x, y, z])


def optimize_distance_for_cluster(
    renderer: Open3DRenderer,
    cluster_points: np.ndarray,
    viewpoint_dir: np.ndarray,
    center: np.ndarray,
    intrinsic,
    image_size: int,
    target_occupancy: float = 0.7,
    min_dist_threshold: float = 0.8
) -> float:
    """优化相机距离以获得最佳视角"""
    # 剔除极端离群点
    dists_to_center = np.linalg.norm(cluster_points - center, axis=1)
    limit_dist = np.percentile(dists_to_center, 99.5)
    core_mask = dists_to_center <= limit_dist
    core_points = cluster_points[core_mask]
    
    total_core_points = len(core_points)
    if total_core_points == 0:
        return max(2.0, min_dist_threshold * 1.5)
    
    max_radius = limit_dist
    fov = 60.0
    min_view_dist = max_radius / np.sin(np.radians(fov / 2.0))
    
    start_dist = max(min_dist_threshold, min_view_dist * 0.6)
    end_dist = max(start_dist * 3.0, min_view_dist * 4.0, 3.5)
    
    test_dists = np.linspace(start_dist, end_dist, 30)
    
    best_dist = end_dist
    best_score = -float('inf')
    
    for dist in test_dists:
        eye = center + viewpoint_dir * dist
        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=center)
        
        pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
            core_points, intrinsic, extrinsic, image_size=image_size
        )
        
        visible_count = np.sum(valid_mask)
        completeness = visible_count / total_core_points
        
        if completeness < 0.995:
            continue
        
        valid_pixels = pixel_coords[valid_mask]
        if len(valid_pixels) > 0:
            min_xy = np.min(valid_pixels, axis=0)
            max_xy = np.max(valid_pixels, axis=0)
            
            margin = image_size * 0.02
            padding_penalty = 0.2 if (min_xy[0] < margin or min_xy[1] < margin or 
                                       max_xy[0] > image_size - margin or 
                                       max_xy[1] > image_size - margin) else 0.0
            
            w = max_xy[0] - min_xy[0]
            h = max_xy[1] - min_xy[1]
            occupancy = (w * h) / (image_size * image_size)
            
            score = 1.0 - abs(occupancy - target_occupancy) - padding_penalty
            
            if score > best_score:
                best_score = score
                best_dist = dist
    
    return best_dist


def get_points_mask(points_2d: np.ndarray, image_size: int, k_size: int = 31) -> np.ndarray:
    """
    从 2D 点生成平滑掩码
    参考 visualize_pointcloud_gradio.py 的实现
    """
    H, W = image_size, image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # 绘制初始点
    for x, y in points_2d:
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(mask, (int(x), int(y)), radius=8, color=255, thickness=-1)
            
    # 闭运算连接空隙
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 高斯模糊 + 阈值化
    mask_blurred = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    _, mask_smooth = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    
    return mask_smooth


def get_robust_mask_from_far_view(
    points: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    current_dist: float,
    intrinsic,
    image_size: int,
    zoom_factor: float = 3.0
) -> np.ndarray:
    """
    利用"远距离视角"生成致密 Mask，然后放大适配当前视角
    解决近距离下点云稀疏导致的 Mask 破碎/颗粒化问题
    参考 visualize_pointcloud_gradio.py 的实现
    """
    # 虚拟拉远相机
    far_dist = current_dist * zoom_factor
    eye_far = center + direction * far_dist
    extrinsic_far = create_camera_extrinsic_from_viewpoint(eye_far, center=center)
    
    # 在远距离下投影
    pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
        points, intrinsic, extrinsic_far, image_size=image_size
    )
    
    valid_pixels = pixel_coords[valid_mask]
    
    if len(valid_pixels) == 0:
        return np.zeros((image_size, image_size), dtype=np.uint8)
        
    # 生成远距离 Mask
    mask_far = get_points_mask(valid_pixels, image_size, k_size=15) 
    
    # 将 Mask 放大 zoom_factor 倍
    H, W = image_size, image_size
    center_x, center_y = W / 2.0, H / 2.0
    
    M = np.array([
        [zoom_factor, 0, (1 - zoom_factor) * center_x],
        [0, zoom_factor, (1 - zoom_factor) * center_y]
    ], dtype=np.float32)
    
    mask_near = cv2.warpAffine(mask_far, M, (W, H), flags=cv2.INTER_LINEAR)
    _, mask_near = cv2.threshold(mask_near, 127, 255, cv2.THRESH_BINARY)
    
    return mask_near


def create_som_overlay_image(
    clean_image: np.ndarray,
    points: np.ndarray,
    child_labels: np.ndarray,
    child_ids: List[int],
    color_map: Dict[int, List[int]],
    viewpoint: np.ndarray,
    center: np.ndarray,
    depth_map: np.ndarray,
    intrinsic,
    image_size: int = 800,
    alpha: float = 0.5,
    dim_background: bool = True,
    draw_contours: bool = True
) -> np.ndarray:
    """
    在 GLB 渲染图上叠加透明颜色蒙版（SoM 风格）
    
    参数:
        clean_image: GLB 渲染的原始图像 (RGB)
        points: 所有点云坐标 (N, 3)
        child_labels: 所有点的聚类标签 (N,)
        child_ids: 兄弟簇的 ID 列表
        color_map: {cluster_id: [R, G, B]} 颜色映射
        viewpoint: 相机位置
        center: 观察中心点
        depth_map: 深度图（用于遮挡检测）
        intrinsic: 相机内参
        image_size: 图像尺寸
        alpha: 蒙版透明度 (0~1, 越大越不透明)
        dim_background: 是否将非簇区域变暗
        draw_contours: 是否绘制轮廓
    
    返回:
        叠加蒙版后的图像
    """
    H, W = image_size, image_size
    extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
    direction = (viewpoint - center)
    direction = direction / np.linalg.norm(direction)
    dist = np.linalg.norm(viewpoint - center)
    
    # 收集所有兄弟簇的点索引
    sibling_mask = np.isin(child_labels, child_ids)
    sibling_points = points[sibling_mask]
    sibling_child_ids = child_labels[sibling_mask]
    
    # 投影所有兄弟簇的点
    pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
        sibling_points, intrinsic, extrinsic, image_size=image_size
    )
    
    # 检查遮挡
    is_visible = np.zeros(len(sibling_points), dtype=bool)
    fov_indices = np.where(valid_mask_fov)[0]
    
    if len(fov_indices) > 0:
        vis_sub = check_visible_points_with_depth(
            sibling_points[fov_indices],
            pixel_coords[fov_indices],
            depths[fov_indices],
            depth_map,
            use_relative_threshold=True,
            relative_threshold_ratio=0.02
        )
        is_visible[fov_indices] = vis_sub
    
    # 为每个子簇生成蒙版
    child_masks = []
    for cid in child_ids:
        c_mask = (sibling_child_ids == cid)
        c_visible = is_visible & c_mask
        c_points_vis = sibling_points[c_visible]
        
        if len(c_points_vis) > 0:
            # 使用远距离视角生成更鲁棒的蒙版
            mask = get_robust_mask_from_far_view(
                c_points_vis, direction, center, dist, intrinsic, image_size, zoom_factor=2.0
            )
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
        
        child_masks.append(mask)
    
    # 合并所有蒙版（用于背景暗化）
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    for mask in child_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 创建结果图像
    result_img = clean_image.copy().astype(np.float32)
    
    # 背景暗化
    if dim_background:
        fg_bool = combined_mask > 0
        result_img[~fg_bool] *= 0.3
    
    # 叠加每个子簇的颜色蒙版
    overlay = result_img.copy()
    for i, (mask, cid) in enumerate(zip(child_masks, child_ids)):
        if mask is None or np.sum(mask) == 0:
            continue
        
        color = np.array(color_map[cid])
        mask_bool = mask > 0
        
        # 叠加半透明颜色
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
        
        # 绘制轮廓
        if draw_contours:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            smooth_contours = []
            for cnt in contours:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                smooth_contours.append(approx)
            cv2.drawContours(overlay, smooth_contours, -1, color.tolist(), 2)
    
    result_img = np.clip(overlay, 0, 255).astype(np.uint8)
    return result_img


def render_pointcloud_som_image(
    points: np.ndarray,
    child_labels: np.ndarray,
    child_ids: List[int],
    color_map: Dict[int, List[int]],
    viewpoint: np.ndarray,
    center: np.ndarray,
    image_size: int = 800,
    point_size: float = 3.0,
    dim_factor: float = 0.25,
    distance: Optional[float] = None
) -> Optional[np.ndarray]:
    """
    渲染点云的 SoM 风格图像
    
    注意：此函数应在 GLB 渲染器清理后调用，避免渲染器资源冲突
    """
    if not OPEN3D_AVAILABLE:
        return None
    
    if distance is None:
        distance = np.linalg.norm(viewpoint - center)
    
    # 距离自适应点大小
    REFERENCE_DIST = 1.5
    scale_factor = (REFERENCE_DIST / max(distance, 0.3)) ** 0.6
    adaptive_point_size = np.clip(point_size * scale_factor, 1.5, 5.0)
    
    # 为每个点分配颜色
    colors = np.zeros((len(points), 3), dtype=np.float64)
    sibling_mask = np.isin(child_labels, child_ids)
    colors[~sibling_mask] = [dim_factor, dim_factor, dim_factor]
    
    for cid in child_ids:
        mask = (child_labels == cid)
        color = np.array(color_map[cid]) / 255.0
        colors[mask] = color
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建渲染器
    renderer = rendering.OffscreenRenderer(image_size, image_size)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = adaptive_point_size * 2
    
    renderer.scene.add_geometry("pointcloud", pcd, mat)
    
    fov = 60.0
    fx = image_size / (2.0 * np.tan(np.radians(fov) / 2.0))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_size, image_size, fx, fx, 
                                                   image_size/2.0, image_size/2.0)
    extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
    
    renderer.setup_camera(intrinsic, extrinsic)
    
    image = renderer.render_to_image()
    img_array = np.asarray(image)[:, :, :3].copy()
    
    # 清理渲染器
    del renderer
    
    return img_array


def find_children_group(
    clustering_results: Dict[int, np.ndarray],
    parent_level_idx: int,
    parent_id: int
) -> Tuple[Optional[int], Optional[np.ndarray], str]:
    """查找父节点的子节点组"""
    parent_labels = clustering_results.get(parent_level_idx)
    if parent_labels is None:
        return None, None, "Parent level not found"
    
    parent_indices = np.where(parent_labels == parent_id)[0]
    if len(parent_indices) == 0:
        return None, None, f"Parent ID {parent_id} is empty"
    
    max_level = max(clustering_results.keys())
    
    print(f"    [DEBUG find_children_group] 搜索 Parent {parent_id} (Level {parent_level_idx}) 的子节点...")
    print(f"    [DEBUG find_children_group] 父节点包含 {len(parent_indices)} 个点")
    
    for lvl in range(parent_level_idx + 1, max_level + 1):
        child_labels = clustering_results.get(lvl)
        if child_labels is None:
            print(f"    [DEBUG find_children_group] Level {lvl} 数据不存在")
            break
        
        current_labels = child_labels[parent_indices]
        unique_children = np.unique(current_labels)
        
        # 统计每个子簇的点数
        child_counts = {cid: np.sum(current_labels == cid) for cid in unique_children}
        print(f"    [DEBUG find_children_group] Level {lvl}: 发现 {len(unique_children)} 个子簇")
        print(f"    [DEBUG find_children_group]   子簇 ID 和点数: {child_counts}")
        
        if len(unique_children) > 1:
            return lvl, unique_children, f"Found split at Level {lvl}"
        
        if lvl == max_level:
            return lvl, unique_children, f"Reached bottom Level {lvl}"
    
    return None, None, "No split found"


def generate_sibling_views(
    points: np.ndarray,
    clustering_results: Dict[int, np.ndarray],
    model: Any,
    parent_level_idx: int,
    parent_id: int,
    image_size: int = 800,
    num_views: int = 4
) -> Tuple[List[Dict], Dict[int, int], int, str]:
    """
    为兄弟簇组生成多视角渲染
    
    返回:
        views: 视角数据列表，每个包含 clean_image, som_image, view_info
        som_id_map: cluster_id -> som_id 映射
        child_level_idx: 实际发生分裂的子层级索引（可能跨越多层）
        message: 状态消息
    """
    # 查找子节点（可能跨多层查找）
    child_level_idx, child_ids, msg = find_children_group(
        clustering_results, parent_level_idx, parent_id
    )
    
    if child_ids is None:
        return [], {}, -1, f"无法找到子节点: {msg}"
    
    child_ids = list(child_ids)
    child_labels = clustering_results[child_level_idx]
    
    print(f"    [DEBUG generate_sibling_views] 原始子簇: {child_ids}")
    
    # 按点数排序并创建 SoM ID 映射
    child_point_counts = [(cid, np.sum(child_labels == cid)) for cid in child_ids]
    child_point_counts.sort(key=lambda x: x[1], reverse=True)
    sorted_child_ids = [x[0] for x in child_point_counts]
    som_id_map = {cid: i + 1 for i, cid in enumerate(sorted_child_ids)}
    child_ids = sorted_child_ids
    
    print(f"    [DEBUG generate_sibling_views] 排序后子簇 (按点数降序): {child_ids}")
    print(f"    [DEBUG generate_sibling_views] 子簇点数详情: {child_point_counts}")
    print(f"    [DEBUG generate_sibling_views] SoM ID 映射: {som_id_map}")
    
    # 收集组内所有点
    group_indices = []
    group_point_child_ids = []
    for cid in child_ids:
        indices = np.where(child_labels == cid)[0]
        group_indices.extend(indices)
        group_point_child_ids.extend([cid] * len(indices))
    
    group_points = points[group_indices]
    group_point_child_ids = np.array(group_point_child_ids)
    group_center = np.mean(group_points, axis=0)
    
    # 生成颜色映射 - 使用预定义的高区分度颜色 (20色)
    # 确保颜色之间有足够的视觉区分度
    DISTINCT_COLORS = [
        [255, 0, 0],      # 红色
        [0, 255, 0],      # 绿色
        [0, 0, 255],      # 蓝色
        [255, 255, 0],    # 黄色
        [255, 0, 255],    # 品红色
        [0, 255, 255],    # 青色
        [255, 128, 0],    # 橙色
        [128, 0, 255],    # 蓝紫色
        [0, 255, 128],    # 春绿色
        [255, 0, 128],    # 玫红色
        [128, 255, 0],    # 酸橙色
        [0, 128, 255],    # 蔚蓝色
        [128, 0, 0],      # 深红色
        [0, 128, 0],      # 深绿色
        [0, 0, 128],      # 深蓝色
        [128, 128, 0],    # 橄榄色
        [128, 0, 128],    # 紫色
        [0, 128, 128],    # 蓝绿色
        [165, 42, 42],    # 棕色
        [255, 215, 0],    # 金色
    ]
    
    color_map = {}
    for i, cid in enumerate(child_ids):
        color_map[cid] = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
    
    print(f"    [DEBUG] 颜色映射 (使用高区分度颜色, 共{len(DISTINCT_COLORS)}种):")
    for cid in child_ids:
        som_id = som_id_map[cid]
        color = color_map[cid]
        count = np.sum(child_labels == cid)
        print(f"      - 簇 ID {cid} (SoM ID {som_id}): {count} 点, 颜色 RGB={color}")
    
    # 搜索最佳视角
    renderer = Open3DRenderer(width=256, height=256)
    renderer.setup()
    renderer.upload_model(model)
    
    fov = 60.0
    cam_params = {
        'intrinsic': {
            'width': 256, 'height': 256,
            'fx': 256 / (2.0 * np.tan(np.radians(fov) / 2.0)),
            'fy': 256 / (2.0 * np.tan(np.radians(fov) / 2.0)),
            'cx': 128.0, 'cy': 128.0, 'fov': fov
        }
    }
    intrinsic = create_camera_intrinsic_from_params(cam_params)
    
    view_points = sample_view_points(radius=1.0, partition=3)
    candidates = []
    
    for vp in view_points:
        direction = vp / np.linalg.norm(vp)
        dist = optimize_distance_for_cluster(
            renderer, group_points, direction, group_center, 
            intrinsic, 256, target_occupancy=0.6, min_dist_threshold=0.5
        )
        
        eye = group_center + direction * dist
        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=group_center)
        
        _, depth_map = renderer.render_view(eye, center=group_center, return_depth=True)
        pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
            group_points, intrinsic, extrinsic, image_size=256
        )
        
        is_visible_depth = np.zeros(len(group_points), dtype=bool)
        fov_indices = np.where(valid_mask_fov)[0]
        
        if len(fov_indices) > 0:
            visible_mask_sub = check_visible_points_with_depth(
                group_points[fov_indices],
                pixel_coords[fov_indices],
                depths[fov_indices],
                depth_map,
                use_relative_threshold=True,
                relative_threshold_ratio=0.02
            )
            is_visible_depth[fov_indices] = visible_mask_sub
        
        valid_mask = is_visible_depth
        
        # 计算评分
        child_vis_list = []
        for cid in child_ids:
            c_mask = (group_point_child_ids == cid)
            c_total = np.sum(c_mask)
            c_visible = np.sum(valid_mask & c_mask)
            c_vis_ratio = c_visible / c_total if c_total > 0 else 0
            child_vis_list.append(c_vis_ratio)
        
        min_child_vis = min(child_vis_list) if child_vis_list else 0
        mean_child_vis = np.mean(child_vis_list) if child_vis_list else 0
        overall_vis = np.sum(valid_mask) / len(group_points)
        
        if overall_vis < 0.1:
            continue
        
        score = (min_child_vis * 10.0) + (mean_child_vis * 5.0) + (overall_vis * 3.0)
        
        candidates.append({
            'score': score,
            'direction': direction,
            'dist': dist,
            'eye': eye,
            'stats': {
                'min_child': min_child_vis,
                'mean_child': mean_child_vis,
                'overall': overall_vis,
                'child_vis_detail': child_vis_list
            }
        })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    valid_candidates = candidates

    # # 筛选视角
    # MIN_WORST_CHILD_VIS = 0.15
    # MIN_MEAN_CHILD_VIS = 0.35
    
    # valid_candidates = [c for c in candidates 
    #                    if c['stats']['min_child'] >= MIN_WORST_CHILD_VIS 
    #                    and c['stats']['mean_child'] >= MIN_MEAN_CHILD_VIS]
    
    # if not valid_candidates:
    #     valid_candidates = [c for c in candidates 
    #                       if c['stats']['min_child'] >= 0.08 
    #                       and c['stats']['mean_child'] >= 0.20]
    
    # if not valid_candidates and candidates:
    #     valid_candidates = candidates[:num_views]
    
    # 选择多样化视角
    DISTINCTNESS_THRESHOLD = 0.7
    final_views = []
    for cand in valid_candidates:
        if len(final_views) >= num_views:
            break
        is_distinct = all(np.dot(cand['direction'], s['direction']) <= DISTINCTNESS_THRESHOLD 
                         for s in final_views)
        if is_distinct:
            final_views.append(cand)
    
    renderer.cleanup()
    
    if not final_views:
        return [], som_id_map, child_level_idx, "无法找到有效视角"
    
    # ========== 关键修复：分离 GLB 渲染和点云渲染，避免渲染器冲突 ==========
    # 参考 visualize_pointcloud_gradio.py 中的实现
    
    # Step 1: 先渲染所有 Clean 视图 (GLB) 和 SoM Overlay 视图
    renderer_final = Open3DRenderer(width=image_size, height=image_size)
    renderer_final.setup()
    renderer_final.upload_model(model)
    
    # 准备相机内参（用于 SoM overlay 生成）
    fov_final = 60.0
    fx_final = image_size / (2.0 * np.tan(np.radians(fov_final) / 2.0))
    cam_params_final = {
        'intrinsic': {
            'width': image_size, 'height': image_size,
            'fx': fx_final, 'fy': fx_final,
            'cx': image_size / 2.0, 'cy': image_size / 2.0,
            'fov': fov_final
        }
    }
    intrinsic_final = create_camera_intrinsic_from_params(cam_params_final)
    
    clean_images = []
    som_overlay_images = []  # 新增：在 GLB 渲染图上叠加蒙版的 SoM 图像
    
    for view in final_views:
        # 渲染 Clean 图像和深度图
        clean_img, depth_map = renderer_final.render_view(view['eye'], center=group_center, return_depth=True)
        clean_images.append(clean_img)
        
        # 生成 SoM Overlay 图像（在 clean_img 上叠加透明颜色蒙版）
        som_overlay_img = create_som_overlay_image(
            clean_img, points, child_labels, child_ids, color_map,
            view['eye'], group_center, depth_map, intrinsic_final,
            image_size=image_size, alpha=0.5, dim_background=True, draw_contours=True
        )
        som_overlay_images.append(som_overlay_img)
    
    # Step 2: 清理 GLB 渲染器（重要：必须在点云渲染之前清理）
    renderer_final.cleanup()
    renderer_final = None
    
    # Step 3: 渲染所有 SoM 视图 (点云) - 此时 GLB 渲染器已释放
    output_views = []
    for i, view in enumerate(final_views):
        clean_img = clean_images[i]
        som_overlay_img = som_overlay_images[i]  # 新增
        
        # 渲染 SoM 图像（点云聚类可视化）
        som_img = render_pointcloud_som_image(
            points, child_labels, child_ids, color_map,
            view['eye'], group_center, image_size=image_size,
            point_size=3.0, dim_factor=0.25, distance=view['dist']
        )
        
        # 计算视角描述
        vec = view['eye'] - group_center
        dist = np.linalg.norm(vec)
        elevation = np.degrees(np.arcsin(vec[1] / dist)) if dist > 1e-6 else 0
        azimuth = np.degrees(np.arctan2(vec[2], vec[0]))
        if azimuth < 0:
            azimuth += 360
        
        view_info = {
            'view_idx': i,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': dist,
            'eye': view['eye'].tolist(),
            'center': group_center.tolist(),
            'stats': view['stats']
        }
        
        output_views.append({
            'clean_image': clean_img,
            'som_image': som_img,
            'som_overlay_image': som_overlay_img,  # 新增：GLB 渲染图 + 透明蒙版叠加
            'view_info': view_info,
            'child_ids': child_ids,
            'som_id_map': som_id_map,
            'color_map': color_map
        })
    
    return output_views, som_id_map, child_level_idx, f"生成了 {len(output_views)} 个视角"


def generate_global_views(
    model: Any,
    image_size: int = 800,
    num_views: int = 4
) -> List[np.ndarray]:
    """生成全局多视角渲染"""
    renderer = Open3DRenderer(width=image_size, height=image_size)
    renderer.setup()
    renderer.upload_model(model)
    
    # 预定义视角
    viewpoints = [
        get_viewpoint_from_angles(0, 15, 1.5),    # 正面
        get_viewpoint_from_angles(90, 15, 1.5),   # 侧面
        get_viewpoint_from_angles(180, 15, 1.5),  # 背面
        get_viewpoint_from_angles(45, 45, 1.5),   # 斜上方
    ]
    
    images = []
    for vp in viewpoints[:num_views]:
        img, _ = renderer.render_view(vp, center=np.array([0, 0, 0]))
        images.append(img)
    
    renderer.cleanup()
    return images


# ===================== MLLM 调用函数 =====================

def call_mllm_for_global_caption(
    client: MLLMClient,
    images: List[np.ndarray]
) -> Dict[str, Any]:
    """调用 MLLM 获取全局标注"""
    # 从文件加载 prompt
    global_caption_prompt = get_prompt("global_caption")
    
    # 构建消息
    content = [{"type": "text", "text": global_caption_prompt}]
    
    for i, img in enumerate(images):
        img_b64 = client.encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content.append({"type": "text", "text": f"[视角 {i+1}]"})
    
    messages = [{"role": "user", "content": content}]
    
    print(f"\n{'='*20} MLLM INPUT (Global Caption) {'='*20}", flush=True)
    print(global_caption_prompt, flush=True)
    print(f"[附带 {len(images)} 张多视角图像]", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    response = client.call(messages, enable_thinking=False)
    
    print(f"\n{'='*20} MLLM OUTPUT (Global Caption) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 解析 JSON
    try:
        # 尝试提取 JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result
    except json.JSONDecodeError:
        pass
    
    return {
        "global_name": "未知物体",
        "global_caption": response,
        "main_parts": []
    }


def get_view_direction_description(azimuth: float, elevation: float) -> str:
    """
    根据方位角和仰角生成定性的视角描述
    
    避免使用"左/右"（有歧义），专注于：
    - 正面/背面/斜向/侧面
    - 俯视/平视/仰视
    """
    # 方位角描述
    # 假设 90° (+Z) 为正面，270° (-Z) 为背面
    az = azimuth % 360
    
    if 67.5 <= az < 112.5:
        azimuth_desc = "正面"
    elif 112.5 <= az < 157.5:
        azimuth_desc = "斜向正面"
    elif 157.5 <= az < 202.5:
        azimuth_desc = "侧面"
    elif 202.5 <= az < 247.5:
        azimuth_desc = "斜向背面"
    elif 247.5 <= az < 292.5:
        azimuth_desc = "背面"
    elif 292.5 <= az < 337.5:
        azimuth_desc = "斜向背面"
    elif 337.5 <= az or az < 22.5:
        azimuth_desc = "侧面"
    else:  # 22.5 <= az < 67.5
        azimuth_desc = "斜向正面"
    
    # 仰角描述
    # > 60: 俯视; 30~60: 高角度俯视; -15~30: 平视; < -15: 仰视
    if elevation > 60:
        elev_desc = "俯视"
    elif elevation > 30:
        elev_desc = "高角度俯视"
    elif elevation > -15:
        elev_desc = "平视"
    else:
        elev_desc = "仰视"
    
    return f"{azimuth_desc}，{elev_desc}"


# 全局颜色名称映射
COLOR_NAMES = {
    (255, 0, 0): "红色",
    (0, 255, 0): "绿色",
    (0, 0, 255): "蓝色",
    (255, 255, 0): "黄色",
    (255, 0, 255): "品红色",
    (0, 255, 255): "青色",
    (255, 128, 0): "橙色",
    (128, 0, 255): "蓝紫色",
    (0, 255, 128): "春绿色",
    (255, 0, 128): "玫红色",
    (128, 255, 0): "酸橙色",
    (0, 128, 255): "蔚蓝色",
    (128, 0, 0): "深红色",
    (0, 128, 0): "深绿色",
    (0, 0, 128): "深蓝色",
    (128, 128, 0): "橄榄色",
    (128, 0, 128): "紫色",
    (0, 128, 128): "蓝绿色",
    (165, 42, 42): "棕色",
    (255, 215, 0): "金色",
}


def get_color_name(rgb: List[int]) -> str:
    """获取 RGB 颜色对应的中文名称"""
    return COLOR_NAMES.get(tuple(rgb), f"RGB{tuple(rgb)}")


def get_color_mapping_str(child_ids: List[int], som_id_map: Dict[int, int], color_map: Dict[int, List[int]]) -> str:
    """生成颜色映射说明字符串"""
    color_mapping_lines = []
    for cid in child_ids:
        som_id = som_id_map[cid]
        color_name = get_color_name(color_map[cid])
        color_mapping_lines.append(f"  - **编号 {som_id}**: {color_name} 区域")
    return "\n".join(color_mapping_lines)


def call_mllm_for_multiview_part_naming(
    client: MLLMClient,
    views: List[Dict],
    global_name: str,
    global_caption: str,
    parent_name: str,
    parent_caption: str,
    child_ids: List[int],
    som_id_map: Dict[int, int],
    color_map: Dict[int, List[int]],
    num_rounds: int = 3
) -> Tuple[List[Dict], Dict]:
    """
    调用 MLLM 进行多视角部件命名（三阶段流程，带多轮采样）
    
    流程：
    1. Step 1a: 送入多视角点云 SoM 图像，获取基于几何结构的命名（3轮，每轮 shuffle）
    2. Step 1b: 送入多视角 GLB 叠加蒙版图像，获取基于纹理的命名（3轮，每轮 shuffle）
    3. Step 1c: 送入拼接图，综合所有结果得到最终命名
    
    参数:
        num_rounds: 每个阶段的采样轮数（默认 3）
    
    返回:
        part_names: 部件名称列表
        log_entry: 日志记录
    """
    # 生成颜色映射说明
    color_mapping_str = get_color_mapping_str(child_ids, som_id_map, color_map)
    
    # 收集日志
    log_entry = {
        "type": "multiview_part_naming_3stage_multiround",
        "num_views": len(views),
        "num_rounds": num_rounds,
        "stages": []
    }
    
    # 不同轮次使用不同的 temperature 增加多样性
    temperatures = [0.4,0.8,1.2]
    
    # 格式化命名结果的辅助函数
    def format_naming_result(part_names_list, round_idx=None):
        if not part_names_list:
            return "（无法解析结果）"
        lines = []
        for p in part_names_list:
            lines.append(f"  - 编号 {p.get('som_id', '?')} [{p.get('color', '?')}]: {p.get('name', '未知')}")
        return "\n".join(lines)
    
    # 解析 JSON 响应的辅助函数
    def parse_naming_response(response: str) -> List[Dict]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
                return result.get('part_names', [])
        except (json.JSONDecodeError, KeyError):
            pass
        return []
    
    # ==================== Step 1a: 点云 SoM 命名 (多轮) ====================
    all_pointcloud_results = []  # 存储所有轮次的结果
    
    step1a_prompt_template = get_prompt("step1a_pointcloud_som_naming")
    step1a_prompt = step1a_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        color_mapping=color_mapping_str
    )
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*20} Step 1a 轮次 {round_idx+1}/{num_rounds}: 点云 SoM 命名 {'='*20}", flush=True)
        
        # Shuffle 视图顺序
        shuffled_indices = list(range(len(views)))
        random.shuffle(shuffled_indices)
        shuffled_views = [views[i] for i in shuffled_indices]
        
        # 构建消息：只包含点云 SoM 图像（按 shuffle 后的顺序）
        content_1a = [{"type": "text", "text": step1a_prompt}]
        for i, view in enumerate(shuffled_views):
            som_img = view.get('som_image')
            if som_img is None:
                continue
            img_b64 = client.encode_image(som_img)
            view_direction = get_view_direction_description(
                view['view_info']['azimuth'],
                view['view_info']['elevation']
            )
            content_1a.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            content_1a.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
        
        messages_1a = [{"role": "user", "content": content_1a}]
        
        print(f"[轮次 {round_idx+1}] Temperature: {temperatures[round_idx % len(temperatures)]}", flush=True)
        print(f"[轮次 {round_idx+1}] 视图顺序 (原索引): {shuffled_indices}", flush=True)
        print(f"[附带 {len(shuffled_views)} 张多视角点云 SoM 图像]", flush=True)
        
        response_1a = client.call(
            messages_1a, 
            enable_thinking=False,
            temperature=temperatures[round_idx % len(temperatures)]
        )
        
        print(f"\n{'='*20} Step 1a 轮次 {round_idx+1} OUTPUT {'='*20}", flush=True)
        print(response_1a, flush=True)
        
        # 解析结果
        pointcloud_part_names = parse_naming_response(response_1a)
        all_pointcloud_results.append({
            "round": round_idx + 1,
            "temperature": temperatures[round_idx % len(temperatures)],
            "shuffle_order": shuffled_indices,
            "response": response_1a,
            "parsed_part_names": pointcloud_part_names
        })
        
        log_entry["stages"].append({
            "stage": f"1a_pointcloud_som_round{round_idx+1}",
            "round": round_idx + 1,
            "temperature": temperatures[round_idx % len(temperatures)],
            "shuffle_order": shuffled_indices,
            "prompt": step1a_prompt,
            "response": response_1a,
            "parsed_part_names": pointcloud_part_names
        })
    
    # ==================== Step 1b: GLB 叠加蒙版命名 (多轮) ====================
    all_glb_overlay_results = []  # 存储所有轮次的结果
    
    step1b_prompt_template = get_prompt("step1b_glb_overlay_naming")
    step1b_prompt = step1b_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        color_mapping=color_mapping_str
    )
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*20} Step 1b 轮次 {round_idx+1}/{num_rounds}: GLB 叠加蒙版命名 {'='*20}", flush=True)
        
        # Shuffle 视图顺序
        shuffled_indices = list(range(len(views)))
        random.shuffle(shuffled_indices)
        shuffled_views = [views[i] for i in shuffled_indices]
        
        # 构建消息：只包含 GLB 叠加蒙版图像（按 shuffle 后的顺序）
        content_1b = [{"type": "text", "text": step1b_prompt}]
        for i, view in enumerate(shuffled_views):
            som_overlay_img = view.get('som_overlay_image')
            if som_overlay_img is None:
                continue
            img_b64 = client.encode_image(som_overlay_img)
            view_direction = get_view_direction_description(
                view['view_info']['azimuth'],
                view['view_info']['elevation']
            )
            content_1b.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            content_1b.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
        
        messages_1b = [{"role": "user", "content": content_1b}]
        
        print(f"[轮次 {round_idx+1}] Temperature: {temperatures[round_idx % len(temperatures)]}", flush=True)
        print(f"[轮次 {round_idx+1}] 视图顺序 (原索引): {shuffled_indices}", flush=True)
        print(f"[附带 {len(shuffled_views)} 张多视角 GLB 叠加蒙版图像]", flush=True)
        
        response_1b = client.call(
            messages_1b, 
            enable_thinking=False,
            temperature=temperatures[round_idx % len(temperatures)]
        )
        
        print(f"\n{'='*20} Step 1b 轮次 {round_idx+1} OUTPUT {'='*20}", flush=True)
        print(response_1b, flush=True)
        
        # 解析结果
        glb_overlay_part_names = parse_naming_response(response_1b)
        all_glb_overlay_results.append({
            "round": round_idx + 1,
            "temperature": temperatures[round_idx % len(temperatures)],
            "shuffle_order": shuffled_indices,
            "response": response_1b,
            "parsed_part_names": glb_overlay_part_names
        })
        
        log_entry["stages"].append({
            "stage": f"1b_glb_overlay_round{round_idx+1}",
            "round": round_idx + 1,
            "temperature": temperatures[round_idx % len(temperatures)],
            "shuffle_order": shuffled_indices,
            "prompt": step1b_prompt,
            "response": response_1b,
            "parsed_part_names": glb_overlay_part_names
        })
    
    # ==================== Step 1c: 综合命名 ====================
    print(f"\n{'='*20} Step 1c: 综合所有轮次结果 {'='*20}", flush=True)
    
    # 格式化所有轮次的点云 SoM 结果
    pointcloud_all_rounds_formatted = []
    for r in all_pointcloud_results:
        round_str = f"**轮次 {r['round']}** (Temperature={r['temperature']}):\n"
        round_str += format_naming_result(r['parsed_part_names'])
        pointcloud_all_rounds_formatted.append(round_str)
    pointcloud_naming_formatted = "\n\n".join(pointcloud_all_rounds_formatted) if pointcloud_all_rounds_formatted else "（无法解析结果）"
    
    # 格式化所有轮次的 GLB 叠加蒙版结果
    glb_all_rounds_formatted = []
    for r in all_glb_overlay_results:
        round_str = f"**轮次 {r['round']}** (Temperature={r['temperature']}):\n"
        round_str += format_naming_result(r['parsed_part_names'])
        glb_all_rounds_formatted.append(round_str)
    glb_overlay_naming_formatted = "\n\n".join(glb_all_rounds_formatted) if glb_all_rounds_formatted else "（无法解析结果）"
    
    step1c_prompt_template = get_prompt("step1c_merge_naming")
    step1c_prompt = step1c_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        color_mapping=color_mapping_str,
        pointcloud_naming_result=pointcloud_naming_formatted,
        glb_overlay_naming_result=glb_overlay_naming_formatted
    )
    
    # 构建消息：包含拼接图像（点云 SoM | GLB 叠加蒙版）
    content_1c = [{"type": "text", "text": step1c_prompt}]
    for i, view in enumerate(views):
        som_img = view.get('som_image')
        som_overlay_img = view.get('som_overlay_image')
        
        if som_img is None or som_overlay_img is None:
            continue
        
        # 拼接图像
        combined_img = np.concatenate([som_img, som_overlay_img], axis=1)
        img_b64 = client.encode_image(combined_img)
        view_direction = get_view_direction_description(
            view['view_info']['azimuth'],
            view['view_info']['elevation']
        )
        content_1c.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content_1c.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}] (左: 点云SoM, 右: GLB叠加蒙版)"})
    
    messages_1c = [{"role": "user", "content": content_1c}]
    
    print(step1c_prompt, flush=True)
    print(f"[附带 {len(views)} 张多视角拼接图像]", flush=True)
    print(f"[综合 Step 1a 的 {num_rounds} 轮结果 和 Step 1b 的 {num_rounds} 轮结果]", flush=True)
    
    response_1c = client.call(messages_1c, enable_thinking=False, temperature=0.3)
    
    print(f"\n{'='*20} Step 1c OUTPUT (最终命名) {'='*20}", flush=True)
    print(response_1c, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 解析最终结果
    part_names = parse_naming_response(response_1c)
    
    if part_names:
        log_entry["final_parsed_result"] = {"part_names": part_names}
    else:
        log_entry['parse_error'] = "Failed to parse final response"
        # 如果最终结果解析失败，尝试使用各轮次中最完整的结果
        # 优先使用 GLB 叠加蒙版的结果（纹理信息更丰富）
        for r in all_glb_overlay_results:
            if r['parsed_part_names']:
                part_names = r['parsed_part_names']
                break
        if not part_names:
            for r in all_pointcloud_results:
                if r['parsed_part_names']:
                    part_names = r['parsed_part_names']
                    break
    
    log_entry["stages"].append({
        "stage": "1c_merge",
        "prompt": step1c_prompt,
        "response": response_1c,
        "final_part_names": part_names
    })
    
    # 记录所有轮次的结果摘要
    log_entry["all_pointcloud_results"] = all_pointcloud_results
    log_entry["all_glb_overlay_results"] = all_glb_overlay_results
    
    return part_names, log_entry


def call_mllm_for_multiview_part_captioning(
    client: MLLMClient,
    views: List[Dict],
    global_name: str,
    global_caption: str,
    parent_name: str,
    parent_caption: str,
    part_names: List[Dict],
    child_ids: List[int],
    som_id_map: Dict[int, int],
    color_map: Dict[int, List[int]]
) -> Tuple[List[Dict], Dict]:
    """
    调用 MLLM 进行多视角部件描述（Step 2）
    一次性发送所有视角的拼接图像，综合生成每个部件的详细描述
    
    返回:
        annotations: 部件标注列表
        log_entry: 日志记录
    """
    # 格式化部件名称信息
    part_names_lines = []
    for part in part_names:
        part_names_lines.append(
            f"  - **编号 {part.get('som_id', '?')}**: {part.get('color', '未知颜色')} 区域 → {part.get('name', '未知部件')}"
        )
    part_names_info = "\n".join(part_names_lines) if part_names_lines else "未能识别部件名称"
    
    # 加载 prompt 模板
    step2_prompt_template = get_prompt("step2_multiview_part_captioning")
    step2_prompt = step2_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        part_names_info=part_names_info
    )
    
    # 构建消息内容：文本 + 多张拼接图像 (Clean|SoM)
    content = [{"type": "text", "text": step2_prompt}]
    
    for i, view in enumerate(views):
        if view.get('som_image') is None:
            continue
        
        # 拼接 Clean 和 SoM 图像
        combined_image = np.concatenate([view['clean_image'], view['som_image']], axis=1)
        img_b64 = client.encode_image(combined_image)
        view_direction = get_view_direction_description(
            view['view_info']['azimuth'],
            view['view_info']['elevation']
        )
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
    
    messages = [{"role": "user", "content": content}]
    
    print(f"\n{'='*20} MLLM INPUT (Multiview Part Captioning) {'='*20}", flush=True)
    print(step2_prompt, flush=True)
    print(f"[附带 {len(views)} 张多视角拼接图像]", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 调用 API（需要 thinking 进行详细描述）
    response = client.call(messages, enable_thinking=False)
    
    print(f"\n{'='*20} MLLM OUTPUT (Multiview Part Captioning) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry = {
        "type": "multiview_part_captioning",
        "prompt": step2_prompt,
        "response": response,
        "num_views": len(views),
        "input_part_names": part_names
    }
    
    # 解析结果
    annotations = []
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            annotations = result.get('annotations', [])
            log_entry['parsed_result'] = result
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse multiview part captioning response: {e}", flush=True)
        log_entry['parse_error'] = str(e)
    
    return annotations, log_entry


def call_mllm_for_sibling_annotation(
    client: MLLMClient,
    clean_image: np.ndarray,
    som_image: np.ndarray,
    global_name: str,
    global_caption: str,
    parent_name: str,
    parent_caption: str,
    current_level: int,
    view_info: Dict,
    child_ids: List[int],
    som_id_map: Dict[int, int],
    color_map: Dict[int, List[int]]
) -> Tuple[ViewAnnotationResult, Dict]:
    """
    调用 MLLM 获取单个视角的标注（两步法）：
    - Step 1: 部件识别 - 从 SoM 图像识别每个颜色区域的部件名称
    - Step 2: 部件描述 - 基于部件名称为每个部件生成详细 caption
    """
    
    view_idx = view_info.get('view_idx', '?')
    
    # 生成视角方向描述
    azimuth = view_info.get('azimuth', 0)
    elevation = view_info.get('elevation', 0)
    view_direction = get_view_direction_description(azimuth, elevation)
    
    # 生成颜色映射说明
    color_mapping_str = get_color_mapping_str(child_ids, som_id_map, color_map)
    
    # ================= Step 1: 部件识别 (基于 SoM View) =================
    
    step1_prompt_template = get_prompt("step1_part_naming")
    step1_prompt = step1_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        view_direction=view_direction,
        color_mapping=color_mapping_str
    )
    
    content_step1 = [
        {"type": "text", "text": step1_prompt},
        {"type": "text", "text": "[图像: SoM 视图 - 颜色编码的点云图]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(som_image)}"}}
    ]
    
    messages_step1 = [{"role": "user", "content": content_step1}]
    
    print(f"\n{'='*20} MLLM INPUT (Step 1: Part Naming - View {view_idx}) {'='*20}", flush=True)
    print(step1_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Step 1: 不需要 Thinking
    step1_response = client.call(messages_step1, enable_thinking=False)
    
    print(f"\n{'='*20} MLLM OUTPUT (Step 1) {'='*20}", flush=True)
    print(step1_response, flush=True)
    print(f"{'='*60}\n", flush=True)

    # 解析 Step 1 结果，提取部件名称
    part_names_info = ""
    try:
        json_start = step1_response.find('{')
        json_end = step1_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            step1_result = json.loads(step1_response[json_start:json_end])
            part_names = step1_result.get('part_names', [])
            # 格式化部件名称信息供 Step 2 使用
            part_names_lines = []
            for part in part_names:
                part_names_lines.append(
                    f"  - **编号 {part.get('som_id', '?')}**: {part.get('color', '未知颜色')} 区域 → {part.get('name', '未知部件')}"
                )
            part_names_info = "\n".join(part_names_lines) if part_names_lines else "未能识别部件名称"
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse Step 1 response: {e}", flush=True)
        part_names_info = "未能解析部件名称，请基于图像自行判断"

    # ================= Step 2: 部件描述 (基于 Combined View + Step 1 部件名) =================
    
    step2_prompt_template = get_prompt("step2_part_captioning")
    step2_prompt = step2_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        view_direction=view_direction,
        part_names_info=part_names_info
    )
    
    # Step 2: 发送拼接图 (Clean | SoM)
    combined_image = np.concatenate([clean_image, som_image], axis=1)
    
    content_step2 = [
        {"type": "text", "text": step2_prompt},
        # {"type": "text", "text": "[图像: Combined 视图 - 左侧 Clean (真实纹理), 右侧 SoM (颜色编码)]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(combined_image)}"}}
    ]
    
    messages_step2 = [{"role": "user", "content": content_step2}]
    
    print(f"\n{'='*20} MLLM INPUT (Step 2: Part Captioning - View {view_idx}) {'='*20}", flush=True)
    print(step2_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Step 2: 需要 Thinking（详细描述需要推理）
    step2_response = client.call(messages_step2)
    
    print(f"\n{'='*20} MLLM OUTPUT (Step 2) {'='*20}", flush=True)
    print(step2_response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry = {
        "type": "sibling_view_annotation_2step",
        "view_idx": view_idx,
        "step1_prompt": step1_prompt,
        "step1_response": step1_response,
        "step2_prompt": step2_prompt,
        "step2_response": step2_response,
        "input_images": ["som_image (Step 1)", "combined_image (Step 2)"]
    }
    
    # 解析 Step 2 结果
    try:
        json_start = step2_response.find('{')
        json_end = step2_response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(step2_response[json_start:json_end])
            
            annotations = []
            for ann in result.get('annotations', []):
                annotations.append(ClusterAnnotation(
                    cluster_id=child_ids[ann['som_id'] - 1] if ann['som_id'] <= len(child_ids) else -1,
                    som_id=ann['som_id'],
                    name=ann.get('name', ''),
                    description=ann.get('description', ''),
                    color=ann.get('color', '未指定')
                ))
            
            # 将解析结果加入日志
            log_entry['parsed_result'] = {
                'view_direction': view_direction,
                'annotations': [asdict(a) for a in annotations],
                'notes': result.get('notes', '')
            }
            
            return ViewAnnotationResult(
                view_idx=view_info['view_idx'],
                view_direction=view_direction,
                annotations=annotations
            ), log_entry
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"解析响应失败: {e}")
    
    # 默认返回
    return ViewAnnotationResult(
        view_idx=view_info['view_idx'],
        view_direction=view_direction,
        annotations=[]
    ), log_entry


def merge_multi_view_annotations(
    client: MLLMClient,
    view_results: List[ViewAnnotationResult],
    views: List[Dict],  # 包含各视角的 clean_image 和 som_image
    global_name: str,
    parent_name: str,
    child_ids: List[int],
    som_id_map: Dict[int, int]
) -> Tuple[List[Dict], Dict]:
    """合并多视角标注结果，综合图像和文本信息选择最优视角"""
    log_entry = {
        "type": "merge_annotations",
        "input_prompt": "",
        "output_response": "",
        "parsed_result": [],
        "input_images": []
    }
    
    if not view_results:
        return [], log_entry
    
    # 格式化多视角标注
    multi_view_str = ""
    for vr in view_results:
        multi_view_str += f"\n### 视角 {vr.view_idx + 1}（{vr.view_direction}）\n"
        multi_view_str += "- 标注结果:\n"
        for ann in vr.annotations:
            multi_view_str += f"  - ID {ann.som_id} [{ann.color}]: {ann.name} - {ann.description}\n"
    
    # 从文件加载 prompt
    merge_prompt_template = get_prompt("merge_annotations")
    prompt = merge_prompt_template.format(
        global_name=global_name,
        parent_name=parent_name,
        num_children=len(child_ids),
        multi_view_annotations=multi_view_str
    )
    
    # 构建消息，包含图像
    content = [{"type": "text", "text": prompt}]
    
    # 添加各视角的拼接图像
    for i, view in enumerate(views):
        if view.get('som_image') is not None:
            # 拼接 clean 和 som 图像
            combined = np.concatenate([view['clean_image'], view['som_image']], axis=1)
            img_b64 = client.encode_image(combined)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            view_direction = get_view_direction_description(
                view['view_info']['azimuth'], 
                view['view_info']['elevation']
            )
            content.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
            log_entry["input_images"].append(f"view_{i+1}_combined")
    
    messages = [{"role": "user", "content": content}]
    
    print(f"\n{'='*20} MLLM INPUT (Merge Annotations) {'='*20}", flush=True)
    print(prompt, flush=True)
    print(f"[附带 {len(log_entry['input_images'])} 张多视角拼接图像]", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Merge: 需要 Thinking
    response = client.call(messages, enable_thinking=False)
    
    print(f"\n{'='*20} MLLM OUTPUT (Merge Annotations) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry["input_prompt"] = prompt
    log_entry["output_response"] = response
    
    # 解析结果
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            merged_list = result.get('merged_annotations', [])
            log_entry["parsed_result"] = merged_list
            return merged_list, log_entry
    except json.JSONDecodeError:
        pass
    
    # 如果解析失败，使用简单的合并策略（取第一个非空描述）
    merged = {}
    for vr in view_results:
        for ann in vr.annotations:
            if ann.som_id not in merged:
                merged[ann.som_id] = {
                    'som_id': ann.som_id,
                    'color': ann.color,
                    'name': ann.name,
                    'caption': ann.description,
                    'best_view': vr.view_direction
                }
            else:
                # 如果当前描述更长，使用当前描述
                if len(ann.description) > len(merged[ann.som_id]['caption']):
                    merged[ann.som_id]['name'] = ann.name
                    merged[ann.som_id]['caption'] = ann.description
                    merged[ann.som_id]['best_view'] = vr.view_direction
    
    result_list = list(merged.values())
    log_entry["parsed_result"] = result_list
    return result_list, log_entry


# ===================== 主流程 =====================

def process_hierarchical_captioning(
    glb_path: str,
    npy_path: str,
    feature_path: str,
    output_dir: str,
    mllm_client: Optional[MLLMClient],
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.3, 0.5, 0.7],
    max_depth: int = 4,
    min_cluster_points: int = 100,
    save_images: bool = True,
    dry_run: bool = False
) -> HierarchicalCaptionResult:
    """
    执行层级化标注的主流程
    """
    start_time = time.time()
    object_id = Path(glb_path).stem
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始层级化标注: {object_id}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据和执行聚类
    print("[Step 1] 加载数据和执行聚类...")
    points, clustering_results, model = load_and_prepare_data(
        glb_path, npy_path, feature_path, k_neighbors, betas
    )
    
    # 2. 全局标注
    print("\n[Step 2] 生成全局标注...")
    global_images = generate_global_views(model, image_size=800, num_views=4)
    
    if save_images:
        for i, img in enumerate(global_images):
            Image.fromarray(img).save(os.path.join(images_dir, f"global_view_{i}.png"))
    
    if dry_run:
        print("  [DRY RUN] 跳过 MLLM 全局标注调用")
        global_name = "调试模式物体"
        global_caption = "这是 dry_run 模式，未调用 MLLM API"
        global_result = {'global_name': global_name, 'global_caption': global_caption, 'confidence': 0.8}
    else:
        # 全局标注开启 thinking
        global_result = call_mllm_for_global_caption(mllm_client, global_images)
        global_name = global_result.get('global_name', '未知物体')
        global_caption = global_result.get('global_caption', '')
    
    print(f"  全局名称: {global_name}")
    print(f"  全局描述: {global_caption[:100]}...")
    
    # 3. 层级遍历和标注
    print("\n[Step 3] 层级遍历和标注...")
    
    # 创建根节点
    root_cluster = ClusterCaption(
        cluster_id=0,
        som_id=0,
        level=0,
        parent_id=None,
        name=global_name,
        caption=global_caption,
        color="",  # 根节点无颜色
        point_count=len(points),
        visible_ratio=1.0,
        children=[]
    )
    
    # BFS 遍历层级树
    queue = deque([(root_cluster, 0, 0)])  # (node, level, cluster_id)
    total_clusters = 1
    max_level_reached = 0
    
    while queue:
        parent_node, parent_level, parent_cluster_id = queue.popleft()
        
        if parent_level >= max_depth:
            continue
        
        print(f"\n  处理: Level {parent_level}, Cluster {parent_cluster_id} ({parent_node.name})")
        
        # 生成兄弟簇视图（可能跨多层查找子节点）
        views, som_id_map, child_level_idx, msg = generate_sibling_views(
            points, clustering_results, model,
            parent_level, parent_cluster_id,
            image_size=800, num_views=4
        )
        
        if not views:
            print(f"    跳过: {msg}")
            continue
        
        # 使用实际的子层级（可能跨越多层）
        child_ids = views[0]['child_ids']
        child_labels = clustering_results.get(child_level_idx)
        
        if child_labels is None:
            print(f"    跳过: 子层级 {child_level_idx} 数据不存在")
            continue
        
        print(f"    发现分裂: Level {parent_level} -> Level {child_level_idx} ({len(child_ids)} 个子簇)")
        
        # ========== 关键调试: 显示所有子簇的详细信息 ==========
        print(f"    [DEBUG] 子簇详情 (min_cluster_points={min_cluster_points}):")
        for cid in child_ids:
            count = np.sum(child_labels == cid)
            status = "✓ 保留" if count >= min_cluster_points else "✗ 被过滤 (点数不足)"
            print(f"      - 簇 ID {cid}: {count} 点 -> {status}")
        
        # 过滤太小的簇
        valid_child_ids = []
        for cid in child_ids:
            count = np.sum(child_labels == cid)
            if count >= min_cluster_points:
                valid_child_ids.append(cid)
        
        print(f"    [DEBUG] 过滤后: {len(child_ids)} -> {len(valid_child_ids)} 个子簇")
        print(f"    [DEBUG] 保留的子簇 ID: {valid_child_ids}")
        
        if len(valid_child_ids) <= 1:
            print(f"    跳过: 有效子簇数量不足 ({len(valid_child_ids)})")
            continue
        
        # 保存视图图像（分别保存各阶段使用的图像）
        if save_images:
            for i, view in enumerate(views):
                som_img = view.get('som_image')  # 点云 SoM
                som_overlay = view.get('som_overlay_image')  # GLB 叠加蒙版
                
                # Step 1a 图像：点云 SoM
                if som_img is not None:
                    Image.fromarray(som_img).save(
                        os.path.join(images_dir, f"L{parent_level}_C{parent_cluster_id}_step1a_pointcloud_{i}.png")
                    )
                
                # Step 1b 图像：GLB 叠加蒙版
                if som_overlay is not None:
                    Image.fromarray(som_overlay).save(
                        os.path.join(images_dir, f"L{parent_level}_C{parent_cluster_id}_step1b_overlay_{i}.png")
                    )
                
                # Step 1c 图像：拼接图（点云SoM | GLB叠加蒙版）
                if som_img is not None and som_overlay is not None:
                    step1c_combined = np.concatenate([som_img, som_overlay], axis=1)
                    Image.fromarray(step1c_combined).save(
                        os.path.join(images_dir, f"L{parent_level}_C{parent_cluster_id}_step1c_combined_{i}.png")
                    )
                
                # Step 2 图像：拼接图（Clean | 点云SoM）
                if som_img is not None:
                    combined = np.concatenate([view['clean_image'], som_img], axis=1)
                    Image.fromarray(combined).save(
                        os.path.join(images_dir, f"L{parent_level}_C{parent_cluster_id}_step2_combined_{i}.png")
                    )
        
        # 准备日志列表
        interaction_logs = []
        
        # 获取颜色映射
        color_map = views[0]['color_map'] if views else {}
        
        # 多视角标注（新流程：直接送入多视角图像）
        if dry_run:
            print(f"    [DRY RUN] 跳过 MLLM 多视角标注调用")
            # 生成模拟的标注结果
            merged_annotations = []
            for i, cid in enumerate(valid_child_ids):
                som_id = som_id_map.get(cid, i + 1)
                color_name = get_color_name(color_map.get(cid, [0, 0, 0]))
                merged_annotations.append({
                    'som_id': som_id,
                    'color': color_name,
                    'name': f'调试部件_{som_id}',
                    'caption': f'这是 dry_run 模式生成的占位描述 (簇ID={cid})',
                    'best_view': '调试视角'
                })
            print(f"    [DRY RUN] 生成了 {len(merged_annotations)} 个模拟标注")
        else:
            # ========== 新流程：两步多视角标注 ==========
            
            # Step 1: 多视角 SoM 图像 -> 部件命名
            print(f"    [Step 1] 多视角部件命名...")
            part_names, naming_log = call_mllm_for_multiview_part_naming(
                mllm_client,
                views,
                global_name,
                global_caption,
                parent_node.name,
                parent_node.caption,
                valid_child_ids,
                som_id_map,
                color_map
            )
            interaction_logs.append(naming_log)
            print(f"    [Step 1] 识别了 {len(part_names)} 个部件名称")
            
            # Step 2: 多视角拼接图像 -> 部件描述
            print(f"    [Step 2] 多视角部件描述...")
            annotations, captioning_log = call_mllm_for_multiview_part_captioning(
                mllm_client,
                views,
                global_name,
                global_caption,
                parent_node.name,
                parent_node.caption,
                part_names,
                valid_child_ids,
                som_id_map,
                color_map
            )
            interaction_logs.append(captioning_log)
            print(f"    [Step 2] 生成了 {len(annotations)} 个部件描述")
            
            # 转换为 merged_annotations 格式（保持与后续代码兼容）
            merged_annotations = []
            for ann in annotations:
                merged_annotations.append({
                    'som_id': ann.get('som_id', 0),
                    'color': ann.get('color', ''),
                    'name': ann.get('name', ''),
                    'caption': ann.get('description', ''),
                    'best_view': ann.get('best_view', '')
                })
            
            # 保存交互日志
            log_path = os.path.join(output_dir, f"L{parent_level}_C{parent_cluster_id}_interaction.json")
            with open(log_path, 'w', encoding='utf-8') as f:
                # 转换 numpy 类型以确保可序列化
                interaction_logs_safe = convert_numpy_types(interaction_logs)
                json.dump(interaction_logs_safe, f, ensure_ascii=False, indent=2)
        
        # 创建子节点
        for ann in merged_annotations:
            som_id = ann.get('som_id', 0)
            if som_id <= 0 or som_id > len(valid_child_ids):
                continue
            
            cluster_id = valid_child_ids[som_id - 1]
            point_count = np.sum(child_labels == cluster_id)
            
            # 计算可见比例（使用最佳视角）
            best_view_stats = views[0]['view_info']['stats'] if views else {'child_vis_detail': [0]}
            vis_idx = som_id - 1
            visible_ratio = best_view_stats['child_vis_detail'][vis_idx] if vis_idx < len(best_view_stats['child_vis_detail']) else 0
            
            child_node = ClusterCaption(
                cluster_id=cluster_id,
                som_id=som_id,
                level=child_level_idx,  # 使用实际的子层级
                parent_id=parent_cluster_id,
                name=ann.get('name', f'部件{som_id}'),
                caption=ann.get('caption', ''),
                color=ann.get('color', ''),
                point_count=point_count,
                visible_ratio=visible_ratio,
                children=[]
            )
            
            parent_node.children.append(child_node)
            queue.append((child_node, child_level_idx, cluster_id))  # 使用实际的子层级
            total_clusters += 1
            max_level_reached = max(max_level_reached, child_level_idx)
            
            print(f"    + {child_node.name}: {point_count} points")
    
    # 4. 保存结果
    processing_time = time.time() - start_time
    
    result = HierarchicalCaptionResult(
        object_id=object_id,
        global_name=global_name,
        global_caption=global_caption,
        root_cluster=root_cluster,
        total_clusters=total_clusters,
        total_levels=max_level_reached + 1,
        processing_time=processing_time
    )
    
    # 保存 JSON
    def cluster_to_dict(cluster: ClusterCaption) -> Dict:
        d = asdict(cluster)
        d['children'] = [cluster_to_dict(c) for c in cluster.children]
        return convert_numpy_types(d)
    
    output_json = {
        'object_id': result.object_id,
        'global_name': result.global_name,
        'global_caption': result.global_caption,
        'hierarchy': cluster_to_dict(result.root_cluster),
        'statistics': {
            'total_clusters': result.total_clusters,
            'total_levels': result.total_levels,
            'processing_time': result.processing_time
        }
    }
    
    json_path = os.path.join(output_dir, f"{object_id}_caption.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"标注完成!")
    print(f"  总簇数: {total_clusters}")
    print(f"  总层级: {max_level_reached + 1}")
    print(f"  处理时间: {processing_time:.1f}s")
    print(f"  结果保存至: {json_path}")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='层级化 3D 物体标注')
    
    # 输入文件
    parser.add_argument('--glb_path', type=str, default='example_material/glbs/e85ebb729b02402bbe3b917e1196f8d3.glb',
                       help='GLB 文件路径')
    parser.add_argument('--npy_path', type=str, default='example_material/npys/e85ebb729b02402bbe3b917e1196f8d3_8192.npy',
                       help='点云 NPY 文件路径')
    parser.add_argument('--feature_path', type=str, default=None,
                       help='特征 NPY 文件路径（如果未指定，将自动推断）')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='outputs/captions',
                       help='输出目录')
    parser.add_argument('--save_images', action='store_true', default=True,
                       help='是否保存中间图像')
    
    # MLLM 配置
    parser.add_argument('--mllm_provider', type=str, default='dashscope',
                       choices=['openai', 'anthropic', 'openai-compatible', 'dashscope'],
                       help='MLLM 提供商 (默认: dashscope)')
    parser.add_argument('--mllm_api_key', type=str, default="sk-7a4e2ece8871495895a9c6a506715e9b",
                       help='MLLM API Key（也可通过环境变量设置: DASHSCOPE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY）')
    parser.add_argument('--mllm_model', type=str, default=None,
                       help='MLLM 模型名称 (dashscope默认: qwen3-vl-plus)')
    parser.add_argument('--mllm_base_url', type=str, default=None,
                       help='MLLM Base URL（用于自定义端点）')
    parser.add_argument('--enable_thinking', action='store_true', default=False,
                       help='启用思考模式 (仅 dashscope 支持)')
    parser.add_argument('--thinking_budget', type=int, default=4096,
                       help='思考模式的最大 Token 数 (默认: 4096)')
    
    # 聚类参数
    parser.add_argument('--k_neighbors', type=int, default=5,
                       help='KNN 邻居数')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7],
                       help='层级聚类的 Beta 参数')
    
    # 其他参数
    parser.add_argument('--max_depth', type=int, default=4,
                       help='最大层级深度')
    parser.add_argument('--min_cluster_points', type=int, default=100,
                       help='最小簇点数')
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='仅执行聚类和渲染，不调用 MLLM API（用于调试）')
    
    args = parser.parse_args()
    
    # 处理 API Key
    api_key = args.mllm_api_key
    if api_key is None:
        if args.mllm_provider == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
        elif args.mllm_provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif args.mllm_provider == 'dashscope':
            api_key = os.environ.get('DASHSCOPE_API_KEY')
        else:
            api_key = os.environ.get('MLLM_API_KEY')
    
    # 自动推断特征路径
    feature_path = args.feature_path
    if feature_path is None:
        object_id = Path(args.npy_path).stem.replace('_8192', '')
        feature_dir = Path(args.npy_path).parent.parent / 'dino_features'
        feature_path = str(feature_dir / f"{object_id}_features.npy")
    
    # dry_run 模式下不需要 API key
    if args.dry_run:
        print("[DRY RUN 模式] 跳过 MLLM API 调用，仅执行聚类和渲染")
        client = None
    else:
        if api_key is None:
            print("错误: 请提供 API Key（通过 --mllm_api_key 或环境变量）")
            print("  DashScope: DASHSCOPE_API_KEY")
            print("  OpenAI: OPENAI_API_KEY")
            print("  Anthropic: ANTHROPIC_API_KEY")
            print("  或者使用 --dry_run 参数跳过 MLLM 调用")
            sys.exit(1)
        
        # 创建 MLLM 客户端
        client = create_mllm_client(
            args.mllm_provider,
            api_key,
            args.mllm_model,
            args.mllm_base_url,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget
        )
    
    # 执行标注
    result = process_hierarchical_captioning(
        glb_path=args.glb_path,
        npy_path=args.npy_path,
        feature_path=feature_path,
        output_dir=args.output_dir,
        mllm_client=client,
        k_neighbors=args.k_neighbors,
        betas=args.betas,
        max_depth=args.max_depth,
        min_cluster_points=args.min_cluster_points,
        save_images=args.save_images,
        dry_run=args.dry_run
    )
    
    return result


if __name__ == '__main__':
    main()

