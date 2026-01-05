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
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
    # 视角和 bbox 信息（用于评判）
    viewpoints: List[Dict] = field(default_factory=list)  # [{eye, center, azimuth, elevation, distance}, ...]
    bboxes: List[Optional[List[int]]] = field(default_factory=list)  # [[x1,y1,x2,y2], ...] 归一化坐标 0-1000
    bbox_3d: Optional[List[float]] = None  # [x_min, y_min, z_min, x_max, y_max, z_max] 3D 包围盒


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
        
        # 生成随机 seed（用于控制输出的可复现性，同时增加多样性）
        seed = random.randint(0, 2**31 - 1)
        
        # 构建请求参数
        kwargs_api = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # 使用流式以支持 thinking
            "temperature": temperature,
            "seed": seed,
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
    betas: List[float] = [0.0, 0.2, 0.35, 0.5]
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Any, np.ndarray]:
    """
    加载数据并执行聚类
    
    返回:
        points: 归一化后的点云坐标
        clustering_results: 各层级的聚类标签
        model: 归一化后的 GLB 模型
        features: 点云特征
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
    
    return points, clustering_results, model, features


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


def get_judge_style_mask(points_2d: np.ndarray, image_size: int) -> np.ndarray:
    """
    使用 judge_2d_bbox.py 中的逻辑生成 Mask
    直接投影 + 形态学闭运算 + 高斯模糊
    """
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 绘制可见点
    for x, y in points_2d:
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(mask, (int(x), int(y)), radius=8, color=255, thickness=-1)
            
    # 形态学操作：闭运算连接空隙
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 高斯模糊 + 阈值化
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask_blurred, 50, 255, cv2.THRESH_BINARY)
    
    return mask


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
    alpha: float = 0.5
) -> np.ndarray:
    """
    在 GLB 渲染图上叠加透明颜色蒙版（SoM 风格）
    完全采用评判脚本风格：仅叠加半透明颜色，无轮廓，无背景压暗
    
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
    
    返回:
        叠加蒙版后的图像
    """
    H, W = image_size, image_size
    extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
    # direction = (viewpoint - center)
    # direction = direction / np.linalg.norm(direction)
    # dist = np.linalg.norm(viewpoint - center)
    
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
        
        # 获取可见点的 2D 坐标
        c_pixels_vis = pixel_coords[c_visible]
        
        if len(c_pixels_vis) > 0:
            # 使用评判脚本的逻辑生成蒙版
            mask = get_judge_style_mask(c_pixels_vis, image_size)
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
        
        child_masks.append(mask)
    
    # 创建结果图像
    result_img = clean_image.copy().astype(np.float32)
    
    # 叠加每个子簇的颜色蒙版 (仅叠加颜色，无背景压暗，无轮廓)
    overlay = result_img.copy()
    for i, (mask, cid) in enumerate(zip(child_masks, child_ids)):
        if mask is None or np.sum(mask) == 0:
            continue
        
        color = np.array(color_map[cid])
        mask_bool = mask > 0
        
        # 叠加半透明颜色
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
        
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


def compute_part_difficulty_weights(
    child_ids: List[int],
    group_features: np.ndarray,
    group_point_child_ids: np.ndarray,
    child_masks_in_group: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """
    计算每个部件的“区分难度权重（混淆度）”，用于“分清”的加权目标。

    精简后的核心逻辑：
    - **部件间越相似（特征中心距离越小）→ 越难分 → 权重越高**
    - 适度加入**部件内部特征分散度**（越分散越难稳定识别）

    返回:
        {cid: weight} 权重归一化到 [1.0, 3.0]
    """
    if len(child_ids) <= 1:
        return {cid: 1.0 for cid in child_ids}

    # 1) 每个部件的特征中心 + 内部分散度
    part_centers: Dict[int, np.ndarray] = {}
    part_spreads: Dict[int, float] = {}
    feat_dim = int(group_features.shape[1]) if group_features.ndim == 2 else 0

    for cid in child_ids:
        mask = child_masks_in_group[cid]
        feats = group_features[mask]
        if len(feats) == 0 or feat_dim == 0:
            part_centers[cid] = np.zeros((feat_dim,), dtype=np.float32)
            part_spreads[cid] = 0.0
            continue
        center = np.mean(feats, axis=0)
        part_centers[cid] = center
        if len(feats) > 1:
            part_spreads[cid] = float(np.mean(np.linalg.norm(feats - center, axis=1)))
        else:
            part_spreads[cid] = 0.0

    # 2) 每个部件找“最近的混淆邻居距离”
    min_dists = []
    for cid_i in child_ids:
        dmin = float("inf")
        for cid_j in child_ids:
            if cid_i == cid_j:
                continue
            d = float(np.linalg.norm(part_centers[cid_i] - part_centers[cid_j]))
            dmin = min(dmin, d)
        min_dists.append(dmin if np.isfinite(dmin) else 0.0)
    min_dists = np.array(min_dists, dtype=np.float32)

    spreads_arr = np.array([part_spreads[cid] for cid in child_ids], dtype=np.float32)

    # 3) 归一化：距离越小→越难；分散度越大→越难
    eps = 1e-6
    if float(min_dists.max() - min_dists.min()) > eps:
        norm_d = (min_dists - float(min_dists.min())) / (float(min_dists.max() - min_dists.min()) + eps)
        confusion = 1.0 - norm_d
    else:
        confusion = np.ones_like(min_dists) * 0.5

    if float(spreads_arr.max() - spreads_arr.min()) > eps:
        norm_s = (spreads_arr - float(spreads_arr.min())) / (float(spreads_arr.max() - spreads_arr.min()) + eps)
    else:
        norm_s = np.zeros_like(spreads_arr)

    # 4) 混淆度为主，分散度为辅（核心逻辑）
    raw = 0.8 * confusion + 0.2 * norm_s

    # 5) 映射到 [1,3]
    # if float(raw.max() - raw.min()) > eps:
    #     raw_n = (raw - float(raw.min())) / (float(raw.max() - raw.min()) + eps)
    #     weights = 1.0 + raw_n * 2.0
    # else:
    # weights = np.ones_like(raw) * 1.5

    return {cid: float(raw[i]) for i, cid in enumerate(child_ids)}


def compute_pair_confusion_scores(
    child_ids: List[int],
    group_features: np.ndarray,
    child_masks_in_group: Dict[int, np.ndarray]
) -> Dict[Tuple[int, int], float]:
    """
    计算“部件对混淆度(confusion)”，用于给“分清(gap)”里的部件对赋权。

    直觉：
    - 两个部件的特征中心越近 -> 越容易混淆 -> 混淆度越高 -> 该对的分离更重要

    返回:
        {(cid_i, cid_j): confusion} 其中 i<j，confusion 归一化到 [0,1]（越大越难分）
    """
    pair_conf: Dict[Tuple[int, int], float] = {}
    if len(child_ids) <= 1:
        return pair_conf

    feat_dim = int(group_features.shape[1]) if group_features.ndim == 2 else 0
    if feat_dim <= 0:
        # 无特征时退化为“等权”
        for ii, cid_i in enumerate(child_ids):
            for jj in range(ii + 1, len(child_ids)):
                pair_conf[(int(cid_i), int(child_ids[jj]))] = 0.5
        return pair_conf

    # 1) 计算每个部件的特征中心
    part_centers: Dict[int, np.ndarray] = {}
    for cid in child_ids:
        mask = child_masks_in_group[cid]
        feats = group_features[mask]
        if len(feats) == 0:
            part_centers[int(cid)] = np.zeros((feat_dim,), dtype=np.float32)
        else:
            part_centers[int(cid)] = np.mean(feats, axis=0)

    # 2) 计算所有部件对的中心距离
    dists: List[float] = []
    pairs: List[Tuple[int, int]] = []
    for ii, cid_i in enumerate(child_ids):
        ci = int(cid_i)
        for jj in range(ii + 1, len(child_ids)):
            cj = int(child_ids[jj])
            d = float(np.linalg.norm(part_centers[ci] - part_centers[cj]))
            dists.append(d)
            pairs.append((ci, cj))

    if not dists:
        return pair_conf

    d_arr = np.array(dists, dtype=np.float32)
    eps = 1e-6
    if float(d_arr.max() - d_arr.min()) > eps:
        norm_d = (d_arr - float(d_arr.min())) / (float(d_arr.max() - d_arr.min()) + eps)
        conf_arr = 1.0 - norm_d
    else:
        conf_arr = np.ones_like(d_arr) * 0.5

    for (ci, cj), c in zip(pairs, conf_arr.tolist()):
        pair_conf[(int(ci), int(cj))] = float(np.clip(c, 0.0, 1.0))
    return pair_conf


def compute_feature_saliency_weights(features: np.ndarray) -> np.ndarray:
    """
    点级别“显著性权重”（看清）：离整体特征中心越远，越“独特/关键”，权重越大。
    输出范围约 [0.5, 3.0]。
    """
    if features is None or len(features) == 0:
        return np.array([], dtype=np.float32)
    feat_center = np.mean(features, axis=0)
    feat_dists = np.linalg.norm(features - feat_center, axis=1)
    eps = 1e-6
    if float(feat_dists.max() - feat_dists.min()) > eps:
        w = (feat_dists - float(feat_dists.min())) / (float(feat_dists.max() - feat_dists.min()) + eps)
        w = 0.5 + w * 1.5
    else:
        w = np.ones_like(feat_dists, dtype=np.float32)
    return w.astype(np.float32)


def compute_mask_gap(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray], min_area: int = 64) -> float:
    """
    计算两个 2D mask 的最小像素间隔 gap（相交则 0）。
    用于“分清”：gap 越大，越容易区分。
    """
    if mask_a is None or mask_b is None:
        return 0.0
    a = (mask_a > 0)
    b = (mask_b > 0)
    if int(a.sum()) < min_area or int(b.sum()) < min_area:
        return 0.0
    if np.logical_and(a, b).any():
        return 0.0
    inv_b = (~b).astype(np.uint8)
    inv_a = (~a).astype(np.uint8)
    try:
        dt_to_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
        dt_to_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 3)
        gap_ab = float(dt_to_b[a].min()) if a.any() else 0.0
        gap_ba = float(dt_to_a[b].min()) if b.any() else 0.0
        return min(gap_ab, gap_ba)
    except Exception:
        return 0.0


def compute_mask_gap_relation(
    mask_a: Optional[np.ndarray],
    mask_b: Optional[np.ndarray],
    min_area: int = 64
) -> Tuple[float, bool]:
    """
    计算两个 2D mask 的最小像素间隔 gap，并显式区分“相交(遮挡/重叠)”。

    返回:
        (gap_px, is_intersect)
        - is_intersect=True: 两 mask 有像素相交，此时 gap_px 固定为 0
        - is_intersect=False: gap_px>=0；gap_px==0 可视为“相邻/贴边”
    """
    if mask_a is None or mask_b is None:
        return 0.0, False
    a = (mask_a > 0)
    b = (mask_b > 0)
    if int(a.sum()) < min_area or int(b.sum()) < min_area:
        return 0.0, False
    if np.logical_and(a, b).any():
        return 0.0, True

    inv_b = (~b).astype(np.uint8)
    inv_a = (~a).astype(np.uint8)
    try:
        dt_to_b = cv2.distanceTransform(inv_b, cv2.DIST_L2, 3)
        dt_to_a = cv2.distanceTransform(inv_a, cv2.DIST_L2, 3)
        gap_ab = float(dt_to_b[a].min()) if a.any() else 0.0
        gap_ba = float(dt_to_a[b].min()) if b.any() else 0.0
        return min(gap_ab, gap_ba), False
    except Exception:
        return 0.0, False


def evaluate_view_candidate(
    renderer: Open3DRenderer,
    group_points: np.ndarray,
    group_point_child_ids: np.ndarray,
    child_ids: List[int],
    intrinsic,
    eye: np.ndarray,
    center: np.ndarray,
    feat_weights: np.ndarray,
    part_difficulty: Dict[int, float],
    pair_confusion: Dict[Tuple[int, int], float],
    image_size: int = 256,
    gap_target_pix: float = 12.0,
    min_mask_area: int = 64
) -> Dict[str, Any]:
    """
    单个候选视角的核心打分（精简版）：
    - **看清**：每个部件的“加权可见性”（显著性权重 feat_weights）
    - **分清**：部件间的 2D gap（难分部件对的 gap 更重要）
    """
    extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=center)
    _, depth_map = renderer.render_view(eye, center=center, return_depth=True)
    pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
        group_points, intrinsic, extrinsic, image_size=image_size
    )

    is_visible = np.zeros(len(group_points), dtype=bool)
    fov_indices = np.where(valid_mask_fov)[0]
    if len(fov_indices) > 0:
        vis_sub = check_visible_points_with_depth(
            group_points[fov_indices],
            pixel_coords[fov_indices],
            depths[fov_indices],
            depth_map,
            use_relative_threshold=True,
            relative_threshold_ratio=0.02
        )
        is_visible[fov_indices] = vis_sub

    child_scores: Dict[int, float] = {}
    child_vis_ratios: List[float] = []
    child_masks_2d_visible: Dict[int, Optional[np.ndarray]] = {}

    for cid in child_ids:
        mask = (group_point_child_ids == cid)
        vmask = is_visible & mask
        weighted_visible = float(np.sum(feat_weights[vmask])) if np.any(vmask) else 0.0
        total_weight = float(np.sum(feat_weights[mask])) if np.any(mask) else 0.0
        score = (weighted_visible / total_weight) if total_weight > 0 else 0.0
        child_scores[cid] = score

        count_visible = int(np.sum(vmask))
        count_total = int(np.sum(mask))
        child_vis_ratios.append((count_visible / count_total) if count_total > 0 else 0.0)

        if count_visible > 0:
            vis_pixels = pixel_coords[vmask]
            child_masks_2d_visible[cid] = get_judge_style_mask(vis_pixels, image_size)
        else:
            child_masks_2d_visible[cid] = None

    # ========= 分清：对每个部件对计算“绝对分离度”（分离 > 相邻 > 相交）=========
    # - 相交(遮挡/重叠)：sep=0
    # - 相邻(贴边，gap≈0)：sep 给予一个小的正基线，确保“相邻 > 相交”
    # - 分离(gap>0)：sep 随 gap 增大而上升，上限 1
    ADJACENT_BASE_SCORE = 0.15
    ADJACENT_EPS_PX = 1e-6

    separation_score = 0.0
    total_pair_w = 0.0
    pair_gap_scores: List[Tuple[int, int, float]] = []  # (cid_i, cid_j, sep_score in [0,1])
    pair_sep_info: List[Tuple[int, int, float, float, str]] = []  # (cid_i, cid_j, sep_score, gap_px, relation)
    for i, cid_i in enumerate(child_ids):
        mi = child_masks_2d_visible.get(cid_i)
        for j in range(i + 1, len(child_ids)):
            cid_j = child_ids[j]
            mj = child_masks_2d_visible.get(cid_j)
            gap_px, is_intersect = compute_mask_gap_relation(mi, mj, min_area=min_mask_area)
            if is_intersect:
                sep_score = 0.0
                relation = "intersect"
            else:
                # 非相交：相邻(gap=0) 也应比相交更好
                sep_raw = float(np.clip(gap_px / gap_target_pix, 0.0, 1.0))
                sep_score = max(ADJACENT_BASE_SCORE, sep_raw)
                relation = "adjacent" if gap_px <= ADJACENT_EPS_PX else "separated"

            pair_gap_scores.append((int(cid_i), int(cid_j), float(sep_score)))
            pair_sep_info.append((int(cid_i), int(cid_j), float(sep_score), float(gap_px), str(relation)))
            # 对的权重应来自“该对的混淆度”，而不是两个部件难度相乘
            key = (int(cid_i), int(cid_j))
            w = float(pair_confusion.get(key, 0.5))
            separation_score += float(sep_score) * w
            total_pair_w += w
    separation_score = (separation_score / total_pair_w) if total_pair_w > 1e-9 else 0.0

    overall_vis = float(np.sum(is_visible) / max(1, len(group_points)))

    return {
        "child_scores": child_scores,
        "vis_ratios": child_vis_ratios,
        "overall_vis": overall_vis,
        "separation_score": float(separation_score),
        "pair_gap_scores": pair_gap_scores,
        "pair_sep_info": pair_sep_info,
    }


def generate_sibling_views(
    points: np.ndarray,
    clustering_results: Dict[int, np.ndarray],
    model: Any,
    parent_level_idx: int,
    parent_id: int,
    features: np.ndarray,
    image_size: int = 800,
    num_views: int = 4
) -> Tuple[List[Dict], Dict[int, int], int, str]:
    """
    为兄弟簇组生成多视角渲染
    
    改进逻辑：
    1. 使用 3D DINO 特征计算点的重要性权重 (Saliency)
    2. 计算部件级别的"区分难度权重"（难分的部件需要更高覆盖）
    3. 使用贪婪算法选择视角组合，最大化所有子部件的加权覆盖率
    
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
    
    # 收集组内所有点
    group_indices = []
    group_point_child_ids = []
    for cid in child_ids:
        indices = np.where(child_labels == cid)[0]
        group_indices.extend(indices)
        group_point_child_ids.extend([cid] * len(indices))
    
    group_points = points[group_indices]
    group_features = features[group_indices]
    group_point_child_ids = np.array(group_point_child_ids)
    group_center = np.mean(group_points, axis=0)
    
    # 预计算每个子簇的 mask (提前计算，供后续多处使用)
    child_masks_in_group = {cid: (group_point_child_ids == cid) for cid in child_ids}
    
    # ========= 精简核心：看清(显著性) + 分清(混淆度权重 + gap) =========
    print(f"    [DEBUG] 计算特征显著性权重(看清) + 部件混淆度权重(难分)...")
    feat_weights = compute_feature_saliency_weights(group_features)
    part_difficulty = compute_part_difficulty_weights(child_ids, group_features, group_point_child_ids, child_masks_in_group)
    pair_confusion = compute_pair_confusion_scores(child_ids, group_features, child_masks_in_group)
    difficulty_info = ", ".join([f"P{som_id_map[cid]}:{part_difficulty.get(cid, 1.0):.2f}" for cid in child_ids])
    if len(feat_weights) > 0:
        print(f"    [DEBUG] 部件难度权重: {difficulty_info}")
        print(f"    [DEBUG] 显著性范围: {float(feat_weights.min()):.2f} - {float(feat_weights.max()):.2f}")
    else:
        print(f"    [DEBUG] 部件难度权重: {difficulty_info}")

    if pair_confusion:
        # 打印最“难分”的若干部件对（混淆度越高越难分）
        top_pair_conf = sorted(pair_confusion.items(), key=lambda x: x[1], reverse=True)[:min(6, len(pair_confusion))]
        pair_conf_info = ", ".join([f"(P{som_id_map[ci]}-P{som_id_map[cj]}:{v:.2f})" for (ci, cj), v in top_pair_conf])
        print(f"    [DEBUG] 部件对混淆度(top): {pair_conf_info}")
    
    # 生成颜色映射
    DISTINCT_COLORS = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], 
        [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 255, 128], [255, 0, 128],
        [128, 255, 0], [0, 128, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128, 128, 0], [128, 0, 128], [0, 128, 128], [165, 42, 42], [255, 215, 0]
    ]
    color_map = {cid: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, cid in enumerate(child_ids)}
    
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
    
    # 使用更密集的采样 (Partition=8 -> ~146 points) 以获得更好覆盖
    view_points = sample_view_points(radius=1.0, partition=4)
    print(f"    [DEBUG] 采样 {len(view_points)} 个候选视角进行覆盖率分析...")
    
    candidates = []
    
    for i, vp in enumerate(view_points):
        direction = vp / np.linalg.norm(vp)
        
        # 优化距离 (快速版，降低分辨率或迭代次数可加速，此处保持默认)
        dist = optimize_distance_for_cluster(
            renderer, group_points, direction, group_center, 
            intrinsic, 256, target_occupancy=0.7, min_dist_threshold=0.5
        )
        
        eye = group_center + direction * dist
        cand_score = evaluate_view_candidate(
            renderer=renderer,
            group_points=group_points,
            group_point_child_ids=group_point_child_ids,
            child_ids=child_ids,
            intrinsic=intrinsic,
            eye=eye,
            center=group_center,
            feat_weights=feat_weights,
            part_difficulty=part_difficulty,
            pair_confusion=pair_confusion,
            image_size=256,
            gap_target_pix=12.0,
            min_mask_area=64
        )
        candidates.append({
            'idx': i,
            'direction': direction,
            'dist': dist,
            'eye': eye,
            **cand_score
        })

    # ========= 精简核心：贪婪选择（看清增益 + 分清gap + 视角多样性）=========
    selected_indices = []
    selected_dirs: List[np.ndarray] = []
    current_coverage = {cid: 0.0 for cid in child_ids}
    # 维护“部件对分离度”的当前最好值：仅用于日志/分析（不再用于增量打分）
    pair_keys: List[Tuple[int, int]] = []
    for ii, cid_i in enumerate(child_ids):
        for jj in range(ii + 1, len(child_ids)):
            pair_keys.append((int(cid_i), int(child_ids[jj])))
    current_pair_sep: Dict[Tuple[int, int], float] = {k: 0.0 for k in pair_keys}
    
    # 分离度得分统计，用于归一化
    all_sep_scores = [c['separation_score'] for c in candidates]
    max_sep = max(all_sep_scores) if all_sep_scores else 1.0
    min_sep = min(all_sep_scores) if all_sep_scores else 0.0
    
    # 权重：分清(gap) + 多样性(方向差异)
    SEPARATION_WEIGHT = 0.25
    DIVERSITY_WEIGHT = 0.15

    # ========= 调试日志控制 =========
    # 打印“每一步里每个候选视角”的各项增益（日志会比较多，默认开启以便排查）
    LOG_ALL_CANDIDATES = True
    LOG_TOP_PAIRS = 6   # 每个候选视角打印贡献最大的部件对数量
    LOG_TOP_PARTS = 6   # 每个候选视角打印增益最大的部件数量
    
    # 增益阈值参数 - 当增益低于阈值时停止添加视角
    MIN_GAIN_RATIO = 0.1      # 相对阈值：增益 < 首视角增益的 10% 时停止
    MIN_GAIN_ABSOLUTE = 0.05  # 绝对阈值：增益 < 0.05 时停止
    
    print(f"    [DEBUG] 开始贪婪选择最多 {num_views} 个视角 (看清增益 + gap分离度 + 多样性)...")
    print(f"    [DEBUG] 分离度范围: {min_sep:.3f} - {max_sep:.3f}")
    print(f"    [DEBUG] 停止条件: 总增益 < max(首视角总增益*{MIN_GAIN_RATIO}, {MIN_GAIN_ABSOLUTE})")
    
    first_view_gain = None  # 记录首视角“总增益”作为基准
    
    for step in range(num_views):
        best_cand_idx = -1
        best_gain = -1.0
        best_vis_gain = 0.0  # 纯看清增益（不含分离度/多样性奖励）
        best_details = {}
        best_sep_bonus = 0.0
        best_div_bonus = 0.0
        best_cand_debug = {}
        
        for i, cand in enumerate(candidates):
            if i in selected_indices:
                continue
                
            # 看清：新视角能为每个部件带来的“显著性加权可见性”提升（难分部件权重大）
            visibility_gain = 0.0
            gain_details = {}
            part_gain_list: List[Tuple[int, float, float]] = []  # (cid, raw_gain, weighted_gain)
            
            for cid in child_ids:
                new_score = cand['child_scores'][cid]
                curr_score = current_coverage[cid]
                # 只有当新视角比已有视角看得更清楚时才计入增益
                raw_gain = max(0, new_score - curr_score)
                
                weighted_gain = float(raw_gain) * float(part_difficulty.get(cid, 1.0))
                visibility_gain += weighted_gain
                gain_details[cid] = (raw_gain, weighted_gain)
                if raw_gain > 0:
                    part_gain_list.append((int(cid), float(raw_gain), float(weighted_gain)))
            
            # 分清：使用“每个视角的绝对分离度”(分离 > 相邻 > 相交) + 混淆度权重
            current_avg_coverage = np.mean(list(current_coverage.values()))
            # sep_importance = 1.0 - current_avg_coverage * 0.5  # 覆盖率高时降低分离度权重
            sep_importance = 1.0
            sep_numer = 0.0
            sep_denom = 0.0
            used_pairs = 0
            pair_contribs: List[Tuple[float, int, int, float, float, float, float, float]] = []
            # (contrib, cid_i, cid_j, sep_val, best_so_far, sep_score, weight, relevance)
            for cid_i, cid_j, sep_score in cand.get("pair_gap_scores", []):
                key = (int(cid_i), int(cid_j))
                best_so_far = float(current_pair_sep.get(key, 0.0))
                sep_val = float(sep_score)

                new_i = float(cand["child_scores"].get(cid_i, 0.0))
                new_j = float(cand["child_scores"].get(cid_j, 0.0))
                curr_i = float(current_coverage.get(cid_i, 0.0))
                curr_j = float(current_coverage.get(cid_j, 0.0))

                pair_vis = min(new_i, new_j)  # 两者都能看见才“分得清”
                # 如果两者在该视角下几乎都看不见，则“分清”的意义不大
                if pair_vis <= 0.3:
                    continue

                # 可见性（而非可见性增量）：越看得见越可靠
                relevance = float(pair_vis)

                # 对的权重应是“该对的混淆度”，而不是两个部件难度相乘
                w = float(pair_confusion.get(key, 0.5))
                weight = w * relevance
                if weight <= 1e-12:
                    continue
                sep_numer += sep_val * weight
                sep_denom += weight
                used_pairs += 1
                pair_contribs.append((
                    float(sep_val * weight),
                    int(cid_i), int(cid_j),
                    float(sep_val), float(best_so_far), float(sep_score),
                    float(weight), float(relevance)
                ))

            # 如果完全没有“可见性对齐”的有效部件对增益，则回退到全局 separation_score 的归一化奖励
            if sep_denom > 1e-9:
                sep_step = sep_numer / sep_denom  # 近似归一化到 [0,1]
            else:
                sep_step = 0
                # if max_sep > min_sep + 1e-6:
                #     sep_step = float((cand["separation_score"] - min_sep) / (max_sep - min_sep))
                # else:
                #     sep_step = 0.5
            separation_bonus = float(sep_step) * SEPARATION_WEIGHT * float(sep_importance)
            
            # 多视角多样性：尽量避免方向重复（与已选方向最大余弦相似越小越好）
            if len(selected_dirs) == 0:
                diversity_score = 1.0
            else:
                sims = [float(np.dot(cand['direction'], d)) for d in selected_dirs]
                max_sim = max(sims) if sims else 1.0
                diversity_score = float(np.clip(1.0 - max_sim, 0.0, 1.0))
            diversity_bonus = diversity_score * DIVERSITY_WEIGHT
            
            # 总增益 = 看清增益 + 分清奖励 + 多样性奖励 + 次要排序键
            total_gain = float(visibility_gain) + float(separation_bonus) + float(diversity_bonus)
            total_gain += float(cand['overall_vis']) * 0.01

            # ========= 候选视角日志：各项增益 + 关键贡献项 =========
            if LOG_ALL_CANDIDATES:
                # 方向与文字描述（与后续 view_info 保持一致的坐标约定）
                d = cand.get("direction", None)
                if d is not None:
                    d = np.asarray(d, dtype=float)
                    d_norm = float(np.linalg.norm(d))
                    if d_norm > 1e-9:
                        dd = d / d_norm
                    else:
                        dd = d
                    cand_el = float(np.degrees(np.arcsin(dd[1]))) if d_norm > 1e-9 else 0.0
                    cand_az = float(np.degrees(np.arctan2(dd[2], dd[0]))) if d_norm > 1e-9 else 0.0
                    if cand_az < 0:
                        cand_az += 360.0
                    cand_dir_desc = get_view_direction_description(cand_az, cand_el)
                    dir_str = f"[{dd[0]:+.2f},{dd[1]:+.2f},{dd[2]:+.2f}]"
                else:
                    cand_az, cand_el = 0.0, 0.0
                    cand_dir_desc = "未知方向"
                    dir_str = "[nan,nan,nan]"

                # 部件增益：按 weighted_gain 降序
                part_gain_list.sort(key=lambda x: x[2], reverse=True)
                top_parts = part_gain_list[:max(0, int(LOG_TOP_PARTS))]
                parts_str = ", ".join([
                    f"P{som_id_map[cid]}:+{raw_g:.2f}(w:+{w_g:.2f})" for cid, raw_g, w_g in top_parts
                ])

                # 部件对贡献：按 contrib 降序
                pair_contribs.sort(key=lambda x: x[0], reverse=True)
                top_pairs = pair_contribs[:max(0, int(LOG_TOP_PAIRS))]
                pairs_str = ", ".join([
                    f"(P{som_id_map[ci]}-P{som_id_map[cj]}:sep{sg:.2f},best{bestv:.2f},conf{wt:.2f},rel{relv:.2f})"
                    for _, ci, cj, sg, bestv, _, wt, relv in top_pairs
                ])

                print(
                    f"      Cand[{i:03d}] "
                    f"Dir={dir_str} (az={cand_az:.1f}, el={cand_el:.1f}, {cand_dir_desc}) | "
                    f"Total={total_gain:.4f} | "
                    f"VisGain={float(visibility_gain):.4f} | "
                    f"SepBonus={float(separation_bonus):.4f} (sep_step={float(sep_step):.3f}, used_pairs={used_pairs}, imp={float(sep_importance):.3f}) | "
                    f"DivBonus={float(diversity_bonus):.4f} | "
                    f"overall_vis={float(cand.get('overall_vis', 0.0)):.3f}"
                )
                if parts_str:
                    print(f"        Parts: {parts_str}")
                if pairs_str:
                    print(f"        Pairs: {pairs_str}")
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_vis_gain = visibility_gain
                best_cand_idx = i
                best_details = gain_details
                best_sep_bonus = separation_bonus
                best_div_bonus = diversity_bonus
                best_cand_debug = {
                    "sep_step": float(sep_step),
                    "sep_importance": float(sep_importance),
                    "used_pairs": int(used_pairs),
                    "azimuth": float(cand_az) if LOG_ALL_CANDIDATES else None,
                    "elevation": float(cand_el) if LOG_ALL_CANDIDATES else None,
                    "dir_desc": str(cand_dir_desc) if LOG_ALL_CANDIDATES else None,
                    "dir_str": str(dir_str) if LOG_ALL_CANDIDATES else None,
                }
        
        if best_cand_idx == -1:
            print(f"      No candidate found, stopping early.")
            break
        
        # 记录首视角总增益作为基准（用于后续提前停止判定）
        if first_view_gain is None:
            first_view_gain = best_gain
            gain_threshold = max(first_view_gain * MIN_GAIN_RATIO, MIN_GAIN_ABSOLUTE)
            print(f"    [DEBUG] 首视角总增益: {first_view_gain:.4f}, 后续停止阈值: {gain_threshold:.4f}")
        else:
            # 检查是否应该停止（第一个视角总是添加）
            gain_threshold = max(first_view_gain * MIN_GAIN_RATIO, MIN_GAIN_ABSOLUTE)
            if best_gain < gain_threshold:
                print(f"      View {step+1} gain ({best_gain:.4f}) < threshold ({gain_threshold:.4f}), stopping early.")
                break
        
        # 添加视角
        selected_indices.append(best_cand_idx)
        cand = candidates[best_cand_idx]
        selected_dirs.append(cand['direction'])
        
        # 更新覆盖率
        print(
            f"      Selected View {step+1}: TotalGain={best_gain:.4f} "
            f"(VisGain={best_vis_gain:.4f}, SepBonus={best_sep_bonus:.4f}, DivBonus={best_div_bonus:.4f}, "
            f"Dir={best_cand_debug.get('dir_str', '[]')} "
            f"(az={best_cand_debug.get('azimuth', 0.0):.1f}, el={best_cand_debug.get('elevation', 0.0):.1f}, "
            f"{best_cand_debug.get('dir_desc', '')}), "
            f"sep_step={best_cand_debug.get('sep_step', 0.0):.3f}, used_pairs={best_cand_debug.get('used_pairs', 0)}, "
            f"sep_imp={best_cand_debug.get('sep_importance', 0.0):.3f})"
        )

        # 打印“部件对分离关系”：分离 > 相邻 > 相交（遮挡/重叠）
        sel_infos = []
        for ci, cj, sep_score, gap_px, rel in cand.get("pair_sep_info", []):
            conf = float(pair_confusion.get((int(ci), int(cj)), 0.5))
            sel_infos.append((conf, float(sep_score), float(gap_px), str(rel), int(ci), int(cj)))
        # 优先看“最混淆”的对
        sel_infos.sort(key=lambda x: x[0], reverse=True)
        top_infos = sel_infos[:max(0, int(LOG_TOP_PAIRS))]
        pair_sep_str = ", ".join([
            f"(P{som_id_map[ci]}-P{som_id_map[cj]}:{rel},gap{gap_px:.1f}px,sep{sep:.2f},conf{conf:.2f})"
            for conf, sep, gap_px, rel, ci, cj in top_infos
        ])
        if pair_sep_str:
            print(f"        PairSep: {pair_sep_str}")

        # 更新部件对分离度（取已选视角中的最大 sep_score）
        for cid_i, cid_j, sep_score in cand.get("pair_gap_scores", []):
            key = (int(cid_i), int(cid_j))
            if key in current_pair_sep and float(sep_score) > float(current_pair_sep[key]):
                current_pair_sep[key] = float(sep_score)
        
        # 打印每个部件的增益明细
        for cid in child_ids:
            score = cand['child_scores'][cid]
            if score > current_coverage[cid]:
                raw_g, weighted_g = best_details.get(cid, (0, 0))
                if raw_g > 0:
                    print(f"        Part {som_id_map[cid]}: +{raw_g:.2f} (weighted: +{weighted_g:.2f}, difficulty: {part_difficulty[cid]:.2f})")
                current_coverage[cid] = score
                
        # 打印当前覆盖状态
        coverage_vals = [f"P{som_id_map[cid]}:{current_coverage[cid]:.2f}" for cid in child_ids]
        print(f"      Current Coverage: {coverage_vals}")
        print(f"      2D Separation: {cand['separation_score']:.3f}")
        
        # 检查是否所有部件都已达到高覆盖率
        min_coverage = min(current_coverage.values())
        if min_coverage >= 0.8:
            print(f"      All parts have >=80% coverage (min={min_coverage:.2f}), stopping early.")
            break
            
    final_views_data = [candidates[i] for i in selected_indices]
    
    renderer.cleanup()
    
    if not final_views_data:
        return [], som_id_map, child_level_idx, "无法找到有效视角"
    
    # ========== 后续渲染逻辑 (复用原有逻辑) ==========
    
    # Step 1: 先渲染所有 Clean 视图 (GLB) 和 SoM Overlay 视图
    renderer_final = Open3DRenderer(width=image_size, height=image_size)
    renderer_final.setup()
    renderer_final.upload_model(model)
    
    fov_final = 60.0
    cam_params_final = {
        'intrinsic': {
            'width': image_size, 'height': image_size,
            'fx': image_size / (2.0 * np.tan(np.radians(fov_final) / 2.0)),
            'fy': image_size / (2.0 * np.tan(np.radians(fov_final) / 2.0)),
            'cx': image_size / 2.0, 'cy': image_size / 2.0,
            'fov': fov_final
        }
    }
    intrinsic_final = create_camera_intrinsic_from_params(cam_params_final)
    
    clean_images = []
    som_overlay_images = []
    
    for cand in final_views_data:
        clean_img, depth_map = renderer_final.render_view(cand['eye'], center=group_center, return_depth=True)
        clean_images.append(clean_img)
        
        som_overlay_img = create_som_overlay_image(
            clean_img, points, child_labels, child_ids, color_map,
            cand['eye'], group_center, depth_map, intrinsic_final,
            image_size=image_size, alpha=0.3
        )
        som_overlay_images.append(som_overlay_img)
    
    renderer_final.cleanup()
    renderer_final = None
    
    # Step 3: 渲染所有 SoM 视图 (点云)
    output_views = []
    for i, cand in enumerate(final_views_data):
        clean_img = clean_images[i]
        som_overlay_img = som_overlay_images[i]
        
        som_img = render_pointcloud_som_image(
            points, child_labels, child_ids, color_map,
            cand['eye'], group_center, image_size=image_size,
            point_size=3.0, dim_factor=0.25, distance=cand['dist']
        )
        
        # 计算视角描述
        vec = cand['eye'] - group_center
        dist = np.linalg.norm(vec)
        elevation = np.degrees(np.arcsin(vec[1] / dist)) if dist > 1e-6 else 0
        azimuth = np.degrees(np.arctan2(vec[2], vec[0]))
        if azimuth < 0: azimuth += 360
        
        # 构造 view_info
        stats = {
            'child_vis_detail': cand['vis_ratios'],
            'overall': cand['overall_vis'],
            'mean_child': np.mean(cand['vis_ratios']) if cand['vis_ratios'] else 0,
            'min_child': min(cand['vis_ratios']) if cand['vis_ratios'] else 0
        }
        
        view_info = {
            'view_idx': i,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': dist,
            'eye': cand['eye'].tolist(),
            'center': group_center.tolist(),
            'stats': stats
        }
        
        # 计算每个子簇的 2D bbox (复用原有逻辑)
        child_bboxes = {}
        # ... (bbox calculation logic repeated here for completeness) ...
        # 为了简洁，这里直接复制原有 BBox 计算逻辑
        
        fov_final = 60.0
        fx_final = image_size / (2.0 * np.tan(np.radians(fov_final) / 2.0))
        cam_params_bbox = {
            'intrinsic': {
                'width': image_size, 'height': image_size,
                'fx': fx_final, 'fy': fx_final,
                'cx': image_size / 2.0, 'cy': image_size / 2.0,
                'fov': fov_final
            }
        }
        intrinsic_bbox = create_camera_intrinsic_from_params(cam_params_bbox)
        extrinsic_bbox = create_camera_extrinsic_from_viewpoint(cand['eye'], center=group_center)
        
        for cid in child_ids:
            c_mask = (child_labels == cid)
            c_points = points[c_mask]
            
            if len(c_points) == 0:
                child_bboxes[cid] = None
                continue
            
            pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
                c_points, intrinsic_bbox, extrinsic_bbox, image_size=image_size
            )
            
            valid_pixels = pixel_coords[valid_mask_fov]
            
            if len(valid_pixels) < 3:
                child_bboxes[cid] = None
                continue
            
            x_min = np.min(valid_pixels[:, 0])
            x_max = np.max(valid_pixels[:, 0])
            y_min = np.min(valid_pixels[:, 1])
            y_max = np.max(valid_pixels[:, 1])
            
            x1 = int(x_min / image_size * 1000)
            y1 = int(y_min / image_size * 1000)
            x2 = int(x_max / image_size * 1000)
            y2 = int(y_max / image_size * 1000)
            
            if x2 > x1 and y2 > y1:
                child_bboxes[cid] = [x1, y1, x2, y2]
            else:
                child_bboxes[cid] = None
        
        output_views.append({
            'clean_image': clean_img,
            'som_image': som_img,
            'som_overlay_image': som_overlay_img,
            'view_info': view_info,
            'child_ids': child_ids,
            'som_id_map': som_id_map,
            'color_map': color_map,
            'child_bboxes': child_bboxes
        })
    
    return output_views, som_id_map, child_level_idx, f"生成了 {len(output_views)} 个优化视角 (Greedy Cover)"


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
    num_rounds: int = 3,
    max_concurrent: int = 8
) -> Tuple[List[Dict], Dict]:
    """
    调用 MLLM 进行多视角部件命名（三阶段流程，带多轮采样）
    
    流程：
    1. Step 1a: 送入多视角点云 SoM 图像，获取基于几何结构的命名（多轮采样）
    2. Step 1b: 送入多视角 GLB 叠加蒙版图像，获取基于纹理的命名（多轮采样）
    3. Step 1c: 送入拼接图，综合所有结果得到最终命名
    
    优化策略：
    - Step 1a 和 Step 1b 的所有轮次并发执行（共 num_rounds * 2 个请求）
    - 同一阶段使用相同消息内容，仅改变 temperature，最大化 prefix cache 命中
    - Step 1c 等待 Step 1a/1b 完成后执行
    
    参数:
        num_rounds: 每个阶段的采样轮数（默认 3）
        max_concurrent: 最大并发请求数（默认 6）
    
    返回:
        part_names: 部件名称列表
        log_entry: 日志记录
    """
    # 生成颜色映射说明
    color_mapping_str = get_color_mapping_str(child_ids, som_id_map, color_map)
    
    # 收集日志
    log_entry = {
        "type": "multiview_part_naming_3stage_concurrent",
        "num_views": len(views),
        "num_rounds": num_rounds,
        "max_concurrent": max_concurrent,
        "stages": []
    }
    
    # 不同轮次使用不同的 temperature 增加多样性
    temperatures = [0.8]
    
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
    
    # ==================== 预先编码图像以复用 (优化 prefix cache) ====================
    som_images_encoded = []  # [(img_b64, view_direction), ...]
    som_overlay_images_encoded = []
    
    for view in views:
        som_img = view.get('som_image')
        som_overlay_img = view.get('som_overlay_image')
        view_direction = get_view_direction_description(
            view['view_info']['azimuth'],
            view['view_info']['elevation']
        )
        
        if som_img is not None:
            som_images_encoded.append((client.encode_image(som_img), view_direction))
        if som_overlay_img is not None:
            som_overlay_images_encoded.append((client.encode_image(som_overlay_img), view_direction))
    
    # ==================== 构建 Step 1a 消息 ====================
    step1a_prompt_template = get_prompt("step1a_pointcloud_som_naming")
    step1a_prompt = step1a_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        color_mapping=color_mapping_str
    )
    
    content_1a = [{"type": "text", "text": step1a_prompt}]
    for i, (img_b64, view_direction) in enumerate(som_images_encoded):
        content_1a.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content_1a.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
    messages_1a = [{"role": "user", "content": content_1a}]
    
    # ==================== 构建 Step 1b 消息 ====================
    step1b_prompt_template = get_prompt("step1b_glb_overlay_naming")
    step1b_prompt = step1b_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        color_mapping=color_mapping_str
    )
    
    content_1b = [{"type": "text", "text": step1b_prompt}]
    for i, (img_b64, view_direction) in enumerate(som_overlay_images_encoded):
        content_1b.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content_1b.append({"type": "text", "text": f"[视角 {i+1}: {view_direction}]"})
    messages_1b = [{"role": "user", "content": content_1b}]
    
    # ==================== 并发执行 Step 1a 和 Step 1b 的所有轮次 ====================
    print(f"\n{'='*20} Step 1a/1b 并发执行 ({num_rounds * 2} 个请求) {'='*20}", flush=True)
    
    def call_step1a(round_idx: int) -> Dict:
        """执行单轮 Step 1a"""
        temp = temperatures[round_idx % len(temperatures)]
        response = client.call(messages_1a, enable_thinking=False, temperature=temp)
        return {
            "stage": "1a",
            "round": round_idx + 1,
            "temperature": temp,
            "response": response,
            "parsed_part_names": parse_naming_response(response)
        }
    
    def call_step1b(round_idx: int) -> Dict:
        """执行单轮 Step 1b"""
        temp = temperatures[round_idx % len(temperatures)]
        response = client.call(messages_1b, enable_thinking=False, temperature=temp)
        return {
            "stage": "1b",
            "round": round_idx + 1,
            "temperature": temp,
            "response": response,
            "parsed_part_names": parse_naming_response(response)
        }
    
    # 使用线程池并发执行
    all_pointcloud_results = []
    all_glb_overlay_results = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # 提交所有任务
        futures_1a = [executor.submit(call_step1a, i) for i in range(num_rounds)]
        futures_1b = [executor.submit(call_step1b, i) for i in range(num_rounds)]
        
        # 收集结果
        for future in futures_1a:
            result = future.result()
            all_pointcloud_results.append(result)
            print(f"  [Step 1a 轮次 {result['round']}] Temperature={result['temperature']}, "
                  f"解析到 {len(result['parsed_part_names'])} 个部件", flush=True)
        
        for future in futures_1b:
            result = future.result()
            all_glb_overlay_results.append(result)
            print(f"  [Step 1b 轮次 {result['round']}] Temperature={result['temperature']}, "
                  f"解析到 {len(result['parsed_part_names'])} 个部件", flush=True)
    
    # 按轮次排序
    all_pointcloud_results.sort(key=lambda x: x['round'])
    all_glb_overlay_results.sort(key=lambda x: x['round'])
    
    # 记录日志
    for r in all_pointcloud_results:
        log_entry["stages"].append({
            "stage": f"1a_pointcloud_som_round{r['round']}",
            "round": r['round'],
            "temperature": r['temperature'],
            "prompt": step1a_prompt,
            "response": r['response'],
            "parsed_part_names": r['parsed_part_names']
        })
    for r in all_glb_overlay_results:
        log_entry["stages"].append({
            "stage": f"1b_glb_overlay_round{r['round']}",
            "round": r['round'],
            "temperature": r['temperature'],
            "prompt": step1b_prompt,
            "response": r['response'],
            "parsed_part_names": r['parsed_part_names']
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


# ===================== 主流程 =====================

def process_hierarchical_captioning(
    glb_path: str,
    npy_path: str,
    feature_path: str,
    output_dir: str,
    mllm_client: Optional[MLLMClient],
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.2, 0.35, 0.5],
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
    points, clustering_results, model, features = load_and_prepare_data(
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
    root_min_bound = np.min(points, axis=0)
    root_max_bound = np.max(points, axis=0)
    root_bbox_3d = root_min_bound.tolist() + root_max_bound.tolist()

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
        children=[],
        bbox_3d=root_bbox_3d
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
            parent_level, parent_cluster_id, features,
            image_size=800, num_views=5
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
            
            # 收集视角和 bbox 信息（用于评判）
            viewpoints_info = []
            bboxes_info = []
            for view in views:
                view_info = view['view_info']
                viewpoints_info.append({
                    'eye': view_info['eye'],
                    'center': view_info['center'],
                    'azimuth': view_info['azimuth'],
                    'elevation': view_info['elevation'],
                    'distance': view_info['distance']
                })
                # 获取该子簇在当前视角的 bbox
                child_bboxes = view.get('child_bboxes', {})
                bboxes_info.append(child_bboxes.get(cluster_id))
            
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
                children=[],
                viewpoints=viewpoints_info,
                bboxes=bboxes_info,
                bbox_3d=None  # 初始化
            )
            
            # 计算 3D bbox
            c_mask = (child_labels == cluster_id)
            c_points = points[c_mask]
            if len(c_points) > 0:
                min_bound = np.min(c_points, axis=0)
                max_bound = np.max(c_points, axis=0)
                # 转换为 list [x_min, y_min, z_min, x_max, y_max, z_max]
                child_node.bbox_3d = min_bound.tolist() + max_bound.tolist()
            
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
    parser.add_argument('--mllm_api_key', type=str, default=None,
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
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.2, 0.35, 0.5],
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

