"""
分层标注评判脚本 - 基于 2D Bounding Box IoU

功能：
1. 读取分层标注结果（JSON 格式）
2. 按兄弟簇组（sibling groups）组织评判
3. 使用与标注时一致的视角
4. 一次性让 MLLM 预测所有兄弟簇的 bbox
5. 计算预测 bbox 与 GT bbox 的 IoU
6. 根据 IoU 阈值判断标注是否正确

使用方法：
    python judge_2d_bbox.py \
        --caption_json outputs/captions/xxx_caption.json \
        --glb_path example_material/glbs/xxx.glb \
        --npy_path example_material/npys/xxx_8192.npy \
        --output_dir outputs/judge_results
"""

import os
import sys
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from io import BytesIO
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

# 添加 Point-R1-data 到 path
sys.path.insert(0, str(Path(__file__).parent / "Point-R1-data"))

from openai import OpenAI
import clustering_utils
from renderer_o3d import (
    Open3DRenderer,
    check_visible_points_with_depth,
    project_points_to_image_with_depth,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint,
    sample_view_points,
)


# ============== 颜色配置 ==============
DISTINCT_COLORS = [
    [255, 0, 0],      # 红色
    [0, 255, 0],      # 绿色
    [0, 0, 255],      # 蓝色
    [255, 255, 0],    # 黄色
    [255, 0, 255],    # 品红色
    [0, 255, 255],    # 青色
    [255, 128, 0],    # 橙色
    [128, 0, 255],    # 蓝紫色
]

COLOR_NAMES = {
    (255, 0, 0): "红色",
    (0, 255, 0): "绿色",
    (0, 0, 255): "蓝色",
    (255, 255, 0): "黄色",
    (255, 0, 255): "品红色",
    (0, 255, 255): "青色",
    (255, 128, 0): "橙色",
    (128, 0, 255): "蓝紫色",
}


# ============== 数据结构 ==============
@dataclass
class PartJudgeResult:
    """单个部件的评判结果"""
    part_name: str
    cluster_id: int
    som_id: int
    level: int
    parent_id: int
    color: str
    gt_mask_points: int  # GT mask 中的点数
    pred_points: List[List[int]]  # 预测的点列表 [[x, y], ...]
    points_in_mask: int  # 落在 GT mask 内的预测点数
    coverage: float  # 覆盖率 = points_in_mask / len(pred_points)
    is_correct: bool
    visible_ratio: float


@dataclass
class SiblingGroupJudgeResult:
    """兄弟簇组的评判结果"""
    parent_name: str
    parent_id: int
    parent_level: int
    child_level: int
    view_idx: int
    num_siblings: int
    part_results: List[PartJudgeResult]
    mean_coverage: float
    accuracy: float


# ============== Prompt 模板 ==============
SIBLING_GROUNDING_PROMPT = """你是一个专业的 3D 物体分析专家。请在下面的 **GLB 3D 模型渲染图** 中找到指定的所有部件位置，并给出每个部件上的 **多个代表性点坐标**。

## 背景信息
- **物体名称**: {object_name}
- **物体描述**: {object_description}
- **当前分析区域**: {parent_name}
- **区域描述**: {parent_caption}

## 需要定位的部件
{parts_info}

## 输出格式
请严格按照以下 JSON 格式输出，为每个部件提供 **2-6 个代表性点**：

```json
{{
    "predictions": [
        {{
            "som_id": 1,
            "label": "部件名称",
            "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
        }},
        ...
    ]
}}
```

其中每个点的坐标范围为 0-1000：
- (0, 0) 为图像左上角
- (1000, 1000) 为图像右下角

## 注意事项
- 点应**均匀分布**在该部件的可见区域内
- 选择能够代表该部件范围的点（包括边缘和中心）
- 如果某个部件在当前视角不可见，`points` 设为空数组 `[]`
- 请为所有列出的部件提供预测

请开始分析：
"""


# ============== DashScope API 客户端 ==============
class DashScopeClient:
    """阿里云 DashScope API 客户端"""
    
    def __init__(self, api_key: str = None, model: str = "qwen3-vl-plus",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def encode_image(self, image: np.ndarray) -> str:
        """将图像编码为 base64"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def call(self, messages: List[Dict], max_tokens: int = 4096,
             stream: bool = False, temperature: float = 0.3) -> str:
        """调用 DashScope API"""
        kwargs_api = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        completion = self.client.chat.completions.create(**kwargs_api)
        return completion.choices[0].message.content


# ============== 覆盖率计算 ==============
def calculate_point_coverage(
    pred_points: List[List[int]],
    gt_mask: np.ndarray,
    image_size: int = 800
) -> Tuple[int, float]:
    """
    计算预测点落在 GT mask 内的覆盖率
    
    Args:
        pred_points: 预测的点列表 [[x, y], ...], 归一化坐标 0-1000
        gt_mask: GT mask (H, W), 值为 255 表示该区域属于部件
        image_size: 图像尺寸
    
    Returns:
        points_in_mask: 落在 mask 内的点数
        coverage: 覆盖率 (0.0 - 1.0)
    """
    if len(pred_points) == 0:
        return 0, 0.0
    
    H, W = gt_mask.shape
    points_in_mask = 0
    
    for point in pred_points:
        # 将归一化坐标转换为像素坐标
        x = int(point[0] / 1000 * W)
        y = int(point[1] / 1000 * H)
        
        # 确保在图像范围内
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        
        # 检查是否在 mask 内
        if gt_mask[y, x] > 127:
            points_in_mask += 1
    
    coverage = points_in_mask / len(pred_points)
    return points_in_mask, coverage


# ============== 点云投影和 Mask 计算 ==============
import cv2

def compute_cluster_gt_mask(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_id: int,
    viewpoint: np.ndarray,
    center: np.ndarray,
    depth_map: np.ndarray,
    image_size: int = 800
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    计算指定聚类簇在给定视角下的 ground truth 2D mask
    
    Returns:
        mask: (H, W) uint8 数组，255 表示该区域属于部件
        visible_ratio: 可见点比例
        mask_points: mask 中的有效像素数
    """
    cluster_mask = cluster_labels == cluster_id
    cluster_points = points[cluster_mask]
    
    if len(cluster_points) == 0:
        return None, 0.0, 0
    
    fov = 60.0
    fx = image_size / (2.0 * np.tan(np.radians(fov) / 2.0))
    cam_params = {
        'intrinsic': {
            'width': image_size, 'height': image_size,
            'fx': fx, 'fy': fx,
            'cx': image_size / 2.0, 'cy': image_size / 2.0,
            'fov': fov
        }
    }
    intrinsic = create_camera_intrinsic_from_params(cam_params)
    extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)
    
    pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
        cluster_points, intrinsic, extrinsic, image_size=image_size
    )
    
    # 检查可见性
    is_visible = np.zeros(len(cluster_points), dtype=bool)
    fov_indices = np.where(valid_mask_fov)[0]
    
    if len(fov_indices) > 0:
        vis_sub = check_visible_points_with_depth(
            cluster_points[fov_indices],
            pixel_coords[fov_indices],
            depths[fov_indices],
            depth_map,
            use_relative_threshold=True,
            relative_threshold_ratio=0.02
        )
        is_visible[fov_indices] = vis_sub
    
    visible_points_2d = pixel_coords[is_visible]
    visible_ratio = np.sum(is_visible) / len(cluster_points)
    
    if len(visible_points_2d) < 3:
        return None, visible_ratio, 0
    
    # 生成 mask
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 绘制可见点
    for x, y in visible_points_2d:
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(mask, (int(x), int(y)), radius=8, color=255, thickness=-1)
    
    # 形态学操作：闭运算连接空隙
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 高斯模糊 + 阈值化
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask_blurred, 50, 255, cv2.THRESH_BINARY)
    
    mask_points = np.sum(mask > 127)
    
    return mask, visible_ratio, mask_points


def optimize_distance_for_cluster(
    renderer: Open3DRenderer,
    cluster_points: np.ndarray,
    viewpoint_dir: np.ndarray,
    center: np.ndarray,
    intrinsic,
    image_size: int,
    target_occupancy: float = 0.6,
    min_dist_threshold: float = 0.5
) -> float:
    """优化相机距离以获得最佳视角（与标注时一致）"""
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


# ============== MLLM 预测 ==============
def predict_sibling_points(
    client: DashScopeClient,
    image: np.ndarray,
    object_name: str,
    object_description: str,
    parent_name: str,
    parent_caption: str,
    siblings: List[Dict],
    verbose: bool = False
) -> Dict[int, List[List[int]]]:
    """
    使用 MLLM 一次性预测所有兄弟簇的代表性点
    
    Returns:
        {som_id: [[x1, y1], [x2, y2], ...]}
    """
    # 构建部件信息
    parts_info_lines = []
    for sibling in siblings:
        som_id = sibling['som_id']
        name = sibling['name']
        caption = sibling['caption']
        color = sibling['color']
        parts_info_lines.append(f"- **编号 {som_id}** ({color}): {name}\n  描述: {caption}")
    parts_info = "\n".join(parts_info_lines)
    
    prompt = SIBLING_GROUNDING_PROMPT.format(
        object_name=object_name,
        object_description=object_description,
        parent_name=parent_name,
        parent_caption=parent_caption,
        parts_info=parts_info
    )
    
    img_b64 = client.encode_image(image)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是一个专业的 3D 物体分析专家，擅长识别 3D 模型的部件位置。"}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    if verbose:
        print(f"    [MLLM] 预测 {len(siblings)} 个兄弟簇的位置点...")
    
    response = client.call(messages, max_tokens=4096, temperature=0.3)
    
    # 解析响应
    predictions = {}
    try:
        text = response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text.strip())
        
        for pred in data.get("predictions", []):
            som_id = pred.get("som_id")
            points = pred.get("points", [])
            if som_id is not None:
                # 确保每个点都是 [x, y] 格式
                valid_points = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        valid_points.append([int(p[0]), int(p[1])])
                predictions[som_id] = valid_points
    except Exception as e:
        if verbose:
            print(f"    [警告] 解析失败: {e}")
            print(f"    原始响应: {response[:500]}...")
    
    return predictions


# ============== 可视化函数 ==============
def visualize_sibling_judge_result(
    image: np.ndarray,
    siblings: List[Dict],
    gt_masks: Dict[int, Optional[np.ndarray]],
    pred_points: Dict[int, List[List[int]]],
    coverages: Dict[int, float],
    output_path: str,
    coverage_threshold: float = 0.5
):
    """
    可视化兄弟簇组的评判结果
    
    - GT mask: 半透明颜色叠加
    - Pred points: 圆点 + 标签显示覆盖率
    """
    img = Image.fromarray(image).convert("RGBA")
    width, height = img.size
    
    # 创建 mask 叠加层
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
        font_small = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size=14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size=12)
        except:
            font = ImageFont.load_default()
            font_small = font
    
    # 绘制 GT masks
    for sibling in siblings:
        som_id = sibling['som_id']
        color_rgb = sibling.get('color_rgb', [255, 0, 0])
        gt_mask = gt_masks.get(som_id)
        
        if gt_mask is not None:
            # 创建半透明 mask 叠加
            mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
            mask_resized = cv2.resize(gt_mask, (width, height))
            mask_bool = mask_resized > 127
            mask_colored[mask_bool] = [color_rgb[0], color_rgb[1], color_rgb[2], 80]  # 半透明
            
            mask_img = Image.fromarray(mask_colored, 'RGBA')
            overlay = Image.alpha_composite(overlay, mask_img)
    
    # 合并 overlay 到原图
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 绘制预测点和标签
    for sibling in siblings:
        som_id = sibling['som_id']
        name = sibling['name']
        color_rgb = sibling.get('color_rgb', [255, 0, 0])
        color = tuple(color_rgb)
        
        points = pred_points.get(som_id, [])
        coverage = coverages.get(som_id, 0.0)
        gt_mask = gt_masks.get(som_id)
        
        # 绘制预测点
        for i, point in enumerate(points):
            px = int(point[0] / 1000 * width)
            py = int(point[1] / 1000 * height)
            
            # 检查点是否在 mask 内
            in_mask = False
            if gt_mask is not None:
                mask_resized = cv2.resize(gt_mask, (width, height))
                px_clamp = max(0, min(px, width - 1))
                py_clamp = max(0, min(py, height - 1))
                in_mask = mask_resized[py_clamp, px_clamp] > 127
            
            # 在 mask 内的点用实心圆，否则用空心圆
            radius = 6
            if in_mask:
                # 实心圆 + 白边
                draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)],
                            fill=color, outline='white', width=2)
            else:
                # 空心圆 (红色边框表示错误)
                draw.ellipse([(px - radius - 1, py - radius - 1), (px + radius + 1, py + radius + 1)],
                            outline='white', width=3)
                draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)],
                            outline=(255, 50, 50), width=2)
        
        # 绘制标签
        if len(points) > 0:
            # 找到点的中心位置放标签
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            label_x = int(avg_x / 1000 * width)
            label_y = int(avg_y / 1000 * height) - 25
            
            is_correct = coverage >= coverage_threshold
            status = "✓" if is_correct else "✗"
            label = f"{name} {status} {coverage:.0%}"
            
            # 带白色背景的标签
            label_bbox = draw.textbbox((label_x, label_y), label, font=font)
            # 确保标签在图像内
            if label_bbox[0] < 0:
                label_x = 5
            if label_bbox[2] > width:
                label_x = width - (label_bbox[2] - label_bbox[0]) - 5
            if label_bbox[1] < 0:
                label_y = 5
            
            label_bbox = draw.textbbox((label_x, label_y), label, font=font)
            draw.rectangle(label_bbox, fill='white')
            label_color = (0, 120, 0) if is_correct else (180, 0, 0)
            draw.text((label_x, label_y), label, fill=label_color, font=font)
    
    # 添加图例
    legend_y = 10
    legend_x = 10
    
    # 背景框
    draw.rectangle([(legend_x - 5, legend_y - 5), (legend_x + 180, legend_y + 55)], fill='white', outline='gray')
    
    draw.text((legend_x, legend_y), "图例:", fill='black', font=font)
    
    # GT mask 图例
    draw.rectangle([(legend_x, legend_y + 18), (legend_x + 20, legend_y + 33)], fill=(255, 0, 0, 80), outline=(255, 0, 0))
    draw.text((legend_x + 25, legend_y + 18), "= GT 区域 (半透明)", fill='black', font=font_small)
    
    # Pred 点图例
    draw.ellipse([(legend_x + 5, legend_y + 38), (legend_x + 15, legend_y + 48)], fill=(255, 0, 0), outline='white')
    draw.text((legend_x + 25, legend_y + 38), "= 预测点", fill='black', font=font_small)
    
    img.save(output_path)


# ============== 提取兄弟簇组 ==============
def extract_sibling_groups(hierarchy: Dict) -> List[Dict]:
    """
    从层级结构中提取所有兄弟簇组
    
    Returns:
        [{
            'parent_name': str,
            'parent_caption': str,
            'parent_id': int,
            'parent_level': int,
            'child_level': int,
            'siblings': [{'som_id', 'cluster_id', 'name', 'caption', 'color', ...}, ...]
        }, ...]
    """
    groups = []
    
    def traverse(node):
        children = node.get('children', [])
        
        if len(children) > 0:
            # 这是一个有子节点的父节点，创建兄弟簇组
            siblings = []
            for child in children:
                siblings.append({
                    'som_id': child.get('som_id', 0),
                    'cluster_id': child.get('cluster_id', -1),
                    'name': child.get('name', '未知'),
                    'caption': child.get('caption', ''),
                    'color': child.get('color', '红色'),
                    'level': child.get('level', 0),
                    'point_count': child.get('point_count', 0),
                    'viewpoints': child.get('viewpoints', []),
                    'bboxes': child.get('bboxes', []),
                })
            
            if len(siblings) > 0:
                # 使用第一个子节点的 level 作为 child_level
                child_level = siblings[0]['level'] if siblings else node.get('level', 0) + 1
                
                groups.append({
                    'parent_name': node.get('name', '未知'),
                    'parent_caption': node.get('caption', ''),
                    'parent_id': node.get('cluster_id', -1),
                    'parent_level': node.get('level', 0),
                    'child_level': child_level,
                    'siblings': siblings,
                })
            
            # 递归处理子节点
            for child in children:
                traverse(child)
    
    traverse(hierarchy)
    return groups


# ============== 主评判函数 ==============
def judge_hierarchical_caption(
    caption_json_path: str,
    glb_path: str,
    npy_path: str,
    feature_path: str,
    output_dir: str,
    mllm_client: Optional[DashScopeClient] = None,
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.3, 0.5, 0.7],
    coverage_threshold: float = 0.6,
    num_views: int = 4,
    save_visualizations: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    执行分层标注评判（按兄弟簇组，基于点覆盖率）
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"开始评判: {caption_json_path}")
        print(f"{'='*60}\n")
    
    # 1. 读取标注结果
    if verbose:
        print("[Step 1] 读取标注结果...")
    
    with open(caption_json_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    object_name = caption_data.get('global_name', '未知物体')
    object_description = caption_data.get('global_caption', '')
    hierarchy = caption_data.get('hierarchy', {})
    
    # 提取兄弟簇组
    sibling_groups = extract_sibling_groups(hierarchy)
    
    if verbose:
        print(f"  物体名称: {object_name}")
        print(f"  提取到 {len(sibling_groups)} 个兄弟簇组:")
        for g in sibling_groups:
            print(f"    - {g['parent_name']} (Level {g['parent_level']}) -> {len(g['siblings'])} 个子部件")
    
    # 2. 加载点云和执行聚类
    if verbose:
        print("\n[Step 2] 加载点云和聚类...")
    
    points_raw = np.load(npy_path)
    if points_raw.shape[1] >= 3:
        points_raw = points_raw[:, :3]
    
    # 坐标轴对齐和归一化
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
    
    # 加载特征并聚类
    features = np.load(feature_path)
    clustering_results = clustering_utils.perform_hierarchical_clustering(
        points, features, k_neighbors, betas
    )
    
    if verbose:
        print(f"  点云形状: {points.shape}")
        print(f"  聚类层级: {list(clustering_results.keys())}")
    
    # 3. 加载 GLB 模型
    if verbose:
        print("\n[Step 3] 加载 GLB 模型...")
    
    renderer = Open3DRenderer(width=800, height=800)
    renderer.setup()
    model = renderer.load_model(glb_path)
    model, _ = renderer.normalize_model(model)
    renderer.upload_model(model)
    
    # 4. 为每个兄弟簇组执行评判
    if verbose:
        print("\n[Step 4] 执行兄弟簇组评判...")
    
    all_group_results = []
    all_part_results = []
    
    for group_idx, group in enumerate(sibling_groups):
        parent_name = group['parent_name']
        parent_caption = group['parent_caption']
        parent_id = group['parent_id']
        parent_level = group['parent_level']
        child_level = group['child_level']
        siblings = group['siblings']
        
        if verbose:
            print(f"\n  评判兄弟簇组 [{group_idx+1}/{len(sibling_groups)}]: {parent_name}")
            print(f"    父节点 Level {parent_level} -> 子节点 Level {child_level}")
            print(f"    包含 {len(siblings)} 个兄弟簇:")
            for s in siblings:
                print(f"      - {s['name']} (som_id={s['som_id']}, cluster_id={s['cluster_id']})")
        
        # 获取子层级的聚类标签
        cluster_labels = clustering_results.get(child_level)
        if cluster_labels is None:
            if verbose:
                print(f"    [跳过] 层级 {child_level} 的聚类结果不存在")
            continue
        
        # 收集组内所有点
        group_indices = []
        for sibling in siblings:
            indices = np.where(cluster_labels == sibling['cluster_id'])[0]
            group_indices.extend(indices)
        
        if len(group_indices) == 0:
            if verbose:
                print(f"    [跳过] 该组没有点")
            continue
        
        group_points = points[group_indices]
        group_center = np.mean(group_points, axis=0)
        
        # 分配颜色
        for i, sibling in enumerate(siblings):
            sibling['color_rgb'] = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
            sibling['color'] = COLOR_NAMES.get(tuple(sibling['color_rgb']), "红色")
        
        # 生成视角（与标注时使用相同的逻辑）
        # 检查是否有缓存的视角
        cached_viewpoints = siblings[0].get('viewpoints', []) if siblings else []
        
        if len(cached_viewpoints) > 0:
            # 使用缓存的视角
            viewpoints_to_use = []
            for vp_info in cached_viewpoints[:num_views]:
                eye = np.array(vp_info['eye'])
                vp_center = np.array(vp_info.get('center', [0, 0, 0]))
                viewpoints_to_use.append({'eye': eye, 'center': vp_center})
            if verbose:
                print(f"    使用缓存的 {len(viewpoints_to_use)} 个视角")
        else:
            # 重新生成视角（与标注时逻辑一致）
            fov = 60.0
            fx = 256 / (2.0 * np.tan(np.radians(fov) / 2.0))
            cam_params = {
                'intrinsic': {
                    'width': 256, 'height': 256,
                    'fx': fx, 'fy': fx,
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
                candidates.append({'eye': eye, 'center': group_center, 'direction': direction})
            
            # 选择多样化视角
            viewpoints_to_use = []
            DISTINCTNESS_THRESHOLD = 0.7
            for cand in candidates:
                if len(viewpoints_to_use) >= num_views:
                    break
                is_distinct = all(
                    np.dot(cand['direction'], np.array(s['eye']) - np.array(s['center'])) / 
                    (np.linalg.norm(np.array(s['eye']) - np.array(s['center'])) + 1e-6) <= DISTINCTNESS_THRESHOLD
                    for s in viewpoints_to_use
                ) if viewpoints_to_use else True
                if is_distinct:
                    viewpoints_to_use.append(cand)
            
            if verbose:
                print(f"    生成了 {len(viewpoints_to_use)} 个新视角")
        
        # 为每个视角执行评判
        for view_idx, vp_info in enumerate(viewpoints_to_use):
            eye = vp_info['eye']
            view_center = vp_info['center']
            
            # 渲染视图
            img, depth_map = renderer.render_view(eye, center=view_center, return_depth=True)
            
            # 计算每个兄弟簇的 GT mask
            gt_masks = {}
            visible_ratios = {}
            mask_points_count = {}
            for sibling in siblings:
                gt_mask, vis_ratio, mask_pts = compute_cluster_gt_mask(
                    points, cluster_labels, sibling['cluster_id'],
                    eye, view_center, depth_map, image_size=800
                )
                gt_masks[sibling['som_id']] = gt_mask
                visible_ratios[sibling['som_id']] = vis_ratio
                mask_points_count[sibling['som_id']] = mask_pts
            
            if verbose:
                print(f"\n    视角 {view_idx}:")
                for sibling in siblings:
                    som_id = sibling['som_id']
                    gt = gt_masks.get(som_id)
                    vis = visible_ratios.get(som_id, 0)
                    mask_pts = mask_points_count.get(som_id, 0)
                    if gt is not None:
                        print(f"      {sibling['name']}: GT mask = {mask_pts} px, 可见 = {vis:.1%}")
                    else:
                        print(f"      {sibling['name']}: 不可见 (可见 = {vis:.1%})")
            
            # 使用 MLLM 预测点
            pred_points_dict = {}
            if mllm_client is not None:
                pred_points_dict = predict_sibling_points(
                    mllm_client, img,
                    object_name, object_description,
                    parent_name, parent_caption,
                    siblings, verbose=verbose
                )
                if verbose:
                    print(f"      MLLM 预测结果:")
                    for sibling in siblings:
                        som_id = sibling['som_id']
                        pred_pts = pred_points_dict.get(som_id, [])
                        print(f"        {sibling['name']}: {len(pred_pts)} 个预测点")
            
            # 计算覆盖率
            coverages = {}
            part_results = []
            for sibling in siblings:
                som_id = sibling['som_id']
                gt_mask = gt_masks.get(som_id)
                pred_pts = pred_points_dict.get(som_id, [])
                
                # 计算点覆盖率
                if gt_mask is not None and len(pred_pts) > 0:
                    points_in_mask, coverage = calculate_point_coverage(pred_pts, gt_mask, image_size=800)
                else:
                    points_in_mask = 0
                    coverage = 0.0
                
                coverages[som_id] = coverage
                is_correct = coverage >= coverage_threshold
                
                result = PartJudgeResult(
                    part_name=sibling['name'],
                    cluster_id=sibling['cluster_id'],
                    som_id=som_id,
                    level=child_level,
                    parent_id=parent_id,
                    color=sibling['color'],
                    gt_mask_points=mask_points_count.get(som_id, 0),
                    pred_points=pred_pts,
                    points_in_mask=points_in_mask,
                    coverage=coverage,
                    is_correct=is_correct,
                    visible_ratio=visible_ratios.get(som_id, 0)
                )
                part_results.append(result)
                all_part_results.append(result)
                
                if verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"      {sibling['name']}: 覆盖率 = {coverage:.1%} ({points_in_mask}/{len(pred_pts)}) {status}")
            
            # 计算组统计
            valid_coverages = [r.coverage for r in part_results if r.gt_mask_points > 0]
            mean_coverage = np.mean(valid_coverages) if valid_coverages else 0.0
            accuracy = sum(1 for r in part_results if r.is_correct and r.gt_mask_points > 0) / len([r for r in part_results if r.gt_mask_points > 0]) if any(r.gt_mask_points > 0 for r in part_results) else 0.0
            
            group_result = SiblingGroupJudgeResult(
                parent_name=parent_name,
                parent_id=parent_id,
                parent_level=parent_level,
                child_level=child_level,
                view_idx=view_idx,
                num_siblings=len(siblings),
                part_results=part_results,
                mean_coverage=mean_coverage,
                accuracy=accuracy
            )
            all_group_results.append(group_result)
            
            # 保存可视化
            if save_visualizations:
                safe_name = parent_name.replace('/', '_').replace('\\', '_')
                vis_path = os.path.join(images_dir, f"{safe_name}_view{view_idx}_judge.png")
                visualize_sibling_judge_result(img, siblings, gt_masks, pred_points_dict, coverages, vis_path, coverage_threshold)
                
                # 同时保存原始渲染图
                img_path = os.path.join(images_dir, f"{safe_name}_view{view_idx}_render.png")
                Image.fromarray(img).save(img_path)
    
    # 清理渲染器
    renderer.cleanup()
    
    # 5. 汇总结果
    processing_time = time.time() - start_time
    
    all_valid_results = [r for r in all_part_results if r.gt_mask_points > 0]
    overall_mean_coverage = np.mean([r.coverage for r in all_valid_results]) if all_valid_results else 0.0
    overall_accuracy = sum(1 for r in all_valid_results if r.is_correct) / len(all_valid_results) if all_valid_results else 0.0
    
    result_summary = {
        'caption_json': caption_json_path,
        'object_name': object_name,
        'coverage_threshold': coverage_threshold,
        'num_sibling_groups': len(sibling_groups),
        'num_evaluated_parts': len(all_valid_results),
        'overall_mean_coverage': overall_mean_coverage,
        'overall_accuracy': overall_accuracy,
        'processing_time': processing_time,
        'group_results': [
            {
                'parent_name': g.parent_name,
                'parent_id': g.parent_id,
                'view_idx': g.view_idx,
                'mean_coverage': g.mean_coverage,
                'accuracy': g.accuracy,
                'parts': [asdict(p) for p in g.part_results]
            }
            for g in all_group_results
        ]
    }
    
    # 保存结果（转换 numpy 类型）
    def convert_numpy_types(obj):
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
    
    result_summary = convert_numpy_types(result_summary)
    
    result_path = os.path.join(output_dir, "judge_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"评判完成!")
        print(f"  兄弟簇组数: {len(sibling_groups)}")
        print(f"  有效部件数: {len(all_valid_results)}")
        print(f"  整体平均覆盖率: {overall_mean_coverage:.1%}")
        print(f"  整体准确率 (覆盖率>={coverage_threshold:.0%}): {overall_accuracy:.1%}")
        print(f"  处理时间: {processing_time:.1f}s")
        print(f"  结果保存至: {result_path}")
        print(f"{'='*60}\n")
    
    return result_summary


# ============== 命令行接口 ==============
def main():
    parser = argparse.ArgumentParser(description="分层标注评判脚本 - 基于 2D Bounding Box IoU")
    
    parser.add_argument('--caption_json', type=str, required=True,
                        help='标注结果 JSON 文件路径')
    parser.add_argument('--glb_path', type=str, default=None,
                        help='GLB 文件路径')
    parser.add_argument('--npy_path', type=str, default=None,
                        help='点云 NPY 文件路径')
    parser.add_argument('--feature_path', type=str, default=None,
                        help='特征 NPY 文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/judge_results',
                        help='输出目录')
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='保存可视化结果')
    parser.add_argument('--api_key', type=str, default=None,
                        help='MLLM API Key')
    parser.add_argument('--model', type=str, default='qwen3-vl-plus',
                        help='MLLM 模型名称')
    parser.add_argument('--coverage_threshold', type=float, default=0.6,
                        help='点覆盖率阈值（预测点落在GT区域内的比例）')
    parser.add_argument('--num_views', type=int, default=4,
                        help='每个兄弟簇组评判的视角数')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='KNN 邻居数')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7],
                        help='层级聚类的 Beta 参数')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='跳过 MLLM 预测')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='打印详细信息')
    
    args = parser.parse_args()
    
    # 推断路径
    caption_json_path = Path(args.caption_json)
    object_id = caption_json_path.stem.replace('_caption', '')
    
    if args.glb_path is None:
        args.glb_path = f"example_material/glbs/{object_id}.glb"
    if args.npy_path is None:
        args.npy_path = f"example_material/npys/{object_id}_8192.npy"
    if args.feature_path is None:
        args.feature_path = f"example_material/dino_features/{object_id}_features.npy"
    
    # 创建 MLLM 客户端
    mllm_client = None
    if not args.dry_run:
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            mllm_client = DashScopeClient(api_key=api_key, model=args.model)
        else:
            print("[警告] 未设置 API Key，将跳过 MLLM 预测")
    
    result = judge_hierarchical_caption(
        caption_json_path=args.caption_json,
        glb_path=args.glb_path,
        npy_path=args.npy_path,
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        mllm_client=mllm_client,
        k_neighbors=args.k_neighbors,
        betas=args.betas,
        coverage_threshold=args.coverage_threshold,
        num_views=args.num_views,
        save_visualizations=args.save_vis,
        verbose=args.verbose
    )
    
    return result


if __name__ == "__main__":
    main()
