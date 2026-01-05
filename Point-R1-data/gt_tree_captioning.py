"""
PartNeXt 部件树（hierarchyList）逐层命名与 caption 生成脚本。

流程（固定、只支持 PartNeXt）：
1) 读取 PartNeXt `hierarchyList`（字段：name/nodeId/maskId/children）
2) 对树做“单子节点父名下沉”修正：如果某节点只有一个子节点，把父节点名加入子节点候选名
3) 对每个节点生成多视角输入图（左 clean，右红色高亮该节点区域），送给 MLLM 输出：
   - 最合适、具体的部件名（中文）
   - 非结构化 caption（2~5 句，包含视觉/几何特征与功能推断）
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 确保可以从同目录 import
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from renderer_o3d import (  # noqa: E402
    Open3DRenderer,
    create_camera_intrinsic_from_params,
    create_camera_extrinsic_from_viewpoint,
    project_points_to_image_with_depth,
    sample_view_points,
)

try:
    import open3d as o3d
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    rendering = None
    OPEN3D_AVAILABLE = False

from hierarchical_captioning import (  # noqa: E402
    MLLMClient,
    create_mllm_client,
    get_prompt,
)

from part_tree_preprocess import preprocess_part_tree  # noqa: E402

try:
    # PartNeXt 是可选依赖：仓库内路径为 PartNeXt/PartNeXt_lib
    PARTNEXT_ROOT = Path(__file__).resolve().parents[1] / "PartNeXt" / "PartNeXt_lib"
    if PARTNEXT_ROOT.exists() and str(PARTNEXT_ROOT) not in sys.path:
        sys.path.insert(0, str(PARTNEXT_ROOT))
    from partnext import PartNeXtDataset  # type: ignore

    PARTNEXT_AVAILABLE = True
except Exception:
    PARTNEXT_AVAILABLE = False


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    # 旧逻辑遗留：当前脚本不再需要结构化 JSON 解析，保留接口以兼容历史调用（不使用）。
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except Exception:
        return None


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # 去掉 ```json / ``` 代码块外壳
    if t.startswith("```"):
        # 只移除最外层 fence，避免误删内容
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_first_json_value(text: str) -> Optional[Any]:
    """
    从文本中提取第一个 JSON 值（对象或数组）。
    兼容模型输出夹杂前后缀、代码块等情况。
    """
    if not text:
        return None
    t = _strip_code_fences(text)

    def _try_parse_first_balanced_json_object(s: str) -> Optional[Any]:
        """
        更鲁棒的 JSON 提取：从左到右寻找第一个能成功 json.loads 的“配平对象/数组”片段。
        处理常见模型输出：前后缀文字、多个 JSON、以及大段 think 导致存在额外的花括号。
        """
        def scan(open_ch: str, close_ch: str) -> Optional[Any]:
            start_positions = [i for i, ch in enumerate(s) if ch == open_ch]
            for start in start_positions:
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(s)):
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                            continue
                        if ch == "\\":
                            esc = True
                            continue
                        if ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == open_ch:
                            depth += 1
                        elif ch == close_ch:
                            depth -= 1
                            if depth == 0:
                                frag = s[start : i + 1]
                                try:
                                    return json.loads(frag)
                                except Exception:
                                    break  # 该 start 无法解析，换下一个 start
                # 该 start 未闭合或解析失败，继续下一个 start
            return None

        # 优先尝试对象，再尝试数组
        obj = scan("{", "}")
        if obj is not None:
            return obj
        arr = scan("[", "]")
        if arr is not None:
            return arr
        return None

    parsed = _try_parse_first_balanced_json_object(t)
    if parsed is not None:
        return parsed

    return None


def _extract_qas_list_best_effort(text: str) -> List[Dict[str, Any]]:
    """
    尽量从“可能被截断/夹杂前后缀”的模型输出中提取 qas 列表。

    目标场景：
    - 输出严格 JSON 但因长度被截断，缺少尾部的 ]/}
    - 输出包含前后解释文字或代码块 fence

    策略：
    - 先定位 "qas" 后面的数组起始 '['
    - 在数组中从左到右扫描每个可配平的 JSON object（{...}），逐个 json.loads
    - 最后一个不完整对象会被丢弃，但之前完整对象会被保留
    """
    if not text:
        return []
    s = _strip_code_fences(text)
    key_pos = s.find('"qas"')
    if key_pos < 0:
        key_pos = s.find("'qas'")
    if key_pos < 0:
        return []
    # 找到数组开始
    arr_pos = s.find("[", key_pos)
    if arr_pos < 0:
        return []

    out: List[Dict[str, Any]] = []
    i = arr_pos + 1
    n = len(s)
    in_str = False
    esc = False
    depth = 0
    obj_start = -1

    # 扫描并提取数组里的每个对象
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        # not in string
        if ch == '"':
            in_str = True
            i += 1
            continue

        if depth == 0:
            # 寻找下一个对象起点
            if ch == "{":
                obj_start = i
                depth = 1
            elif ch == "]":
                break
            i += 1
            continue

        # depth > 0：正在对象内部配平
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                frag = s[obj_start : i + 1]
                try:
                    obj = json.loads(frag)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    # 单个对象解析失败：忽略并继续
                    pass
                obj_start = -1
        i += 1

    return out


def _as_name_candidates(node: Dict[str, Any]) -> List[str]:
    cands: List[str] = []
    name = node.get("name")
    if isinstance(name, str) and name.strip():
        cands.append(name.strip())
    elif isinstance(name, list):
        for x in name:
            if isinstance(x, str) and x.strip():
                cands.append(x.strip())

    extra = node.get("name_candidates")
    if isinstance(extra, list):
        for x in extra:
            if isinstance(x, str) and x.strip():
                cands.append(x.strip())

    # 去重（保序）
    seen = set()
    out = []
    for x in cands:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _append_name_candidates(node: Dict[str, Any], new_cands: List[str]) -> None:
    """将 new_cands 追加进 node 的 name_candidates（保序去重）。"""
    if not new_cands:
        return
    base = []
    if isinstance(node.get("name_candidates"), list):
        base = [x for x in node.get("name_candidates") if isinstance(x, str) and x.strip()]
    merged = base + [x for x in new_cands if isinstance(x, str) and x.strip()]
    seen = set()
    out = []
    for x in merged:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    if out:
        node["name_candidates"] = out


def propagate_single_child_names_to_child_candidates(root: Dict[str, Any]) -> None:
    """
    按你的流程要求做树修正：
    若某个节点只有一个子节点，则把该节点的名字（及其候选）放入子节点的候选名称列表。
    """
    def rec(node: Dict[str, Any]):
        children = _collect_children(node)
        if len(children) == 1:
            parent_cands = _as_name_candidates(node)
            _append_name_candidates(children[0], parent_cands)
        for c in children:
            rec(c)

    rec(root)


def get_viewpoint_from_angles(azimuth: float, elevation: float, radius: float) -> np.ndarray:
    theta = np.radians(90 - elevation)
    phi = np.radians(azimuth)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z], dtype=np.float32)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    if mask is None or mask.size == 0:
        return image
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image.copy()
    m = mask > 0
    if not np.any(m):
        return img
    overlay = img.astype(np.float32)
    overlay[m] = overlay[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# 高区分度颜色列表（用于区分不同兄弟节点）
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


def render_pointcloud_siblings(
    all_points: np.ndarray,
    sibling_indices_list: List[np.ndarray],
    sibling_colors: List[Tuple[int, int, int]],
    viewpoint: np.ndarray,
    center: np.ndarray,
    image_size: int = 800,
    point_size: float = 3.0,
    dim_factor: float = 0.3,
    pc_renderer: Optional["rendering.OffscreenRenderer"] = None,
) -> Optional[np.ndarray]:
    """
    使用点云方式渲染所有兄弟节点（SoM 风格）：
    - 每个兄弟节点用不同颜色
    - 非兄弟区域用暗灰色
    - 直接渲染点云，不使用 mask 叠加

    参数:
        all_points: 所有点云坐标 (N, 3)
        sibling_indices_list: 每个兄弟节点的点索引列表 [indices_0, indices_1, ...]
        sibling_colors: 每个兄弟节点的颜色 [(R,G,B), ...]
        viewpoint: 相机位置
        center: 观察中心点
        image_size: 图像尺寸
        point_size: 基准点大小
        dim_factor: 非兄弟点的暗化系数
        pc_renderer: 可选的已有渲染器（避免创建多个 OffscreenRenderer 导致冲突）

    返回:
        渲染的图像数组 (H, W, 3) 或 None
    """
    if not OPEN3D_AVAILABLE:
        return None

    if all_points is None or len(all_points) == 0:
        return None

    # 计算相机距离并自适应点大小
    distance = np.linalg.norm(viewpoint - center)
    REFERENCE_DIST = 1.5
    scale_factor = (REFERENCE_DIST / max(distance, 0.3)) ** 0.6
    adaptive_point_size = point_size * scale_factor
    adaptive_point_size = np.clip(adaptive_point_size, 1.5, 6.0)

    # 创建颜色数组 - 默认暗灰色
    colors = np.ones((len(all_points), 3), dtype=np.float64) * dim_factor

    # 为每个兄弟节点分配颜色
    for indices, color in zip(sibling_indices_list, sibling_colors):
        if indices is not None and len(indices) > 0:
            color_normalized = np.array(color) / 255.0
            colors[indices] = color_normalized

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 使用已有渲染器或创建新的
    owns_renderer = False
    if pc_renderer is None:
        pc_renderer = rendering.OffscreenRenderer(image_size, image_size)
        owns_renderer = True
    
    pc_renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    pc_renderer.scene.clear_geometry()

    # 创建点云材质
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = adaptive_point_size * 2

    # 添加点云到场景
    pc_renderer.scene.add_geometry("pointcloud", pcd, mat)

    # 设置相机
    fov = 60.0
    fx = image_size / (2.0 * np.tan(np.radians(fov) / 2.0))
    fy = fx
    cx = image_size / 2.0
    cy = image_size / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_size, image_size, fx, fy, cx, cy)
    extrinsic = create_camera_extrinsic_from_viewpoint(viewpoint, center=center)

    pc_renderer.setup_camera(intrinsic, extrinsic)

    # 渲染
    image = pc_renderer.render_to_image()
    img_array = np.asarray(image)[:, :, :3].copy()

    # 清理场景（但保留渲染器以供复用）
    pc_renderer.scene.clear_geometry()
    
    if owns_renderer:
        del pc_renderer
    
    return img_array


def render_pointcloud_highlight(
    all_points: np.ndarray,
    highlight_indices: np.ndarray,
    viewpoint: np.ndarray,
    center: np.ndarray,
    image_size: int = 800,
    point_size: float = 3.0,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    background_color: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    dim_factor: float = 0.4,
    pc_renderer: Optional["rendering.OffscreenRenderer"] = None,
) -> Optional[np.ndarray]:
    """
    使用点云方式渲染高亮图像（单个节点高亮，兼容旧接口）
    """
    if highlight_indices is None or len(highlight_indices) == 0:
        return None
    return render_pointcloud_siblings(
        all_points=all_points,
        sibling_indices_list=[highlight_indices],
        sibling_colors=[highlight_color],
        viewpoint=viewpoint,
        center=center,
        image_size=image_size,
        point_size=point_size,
        dim_factor=dim_factor,
        pc_renderer=pc_renderer,
    )


def optimize_view_distance(
    region_points: np.ndarray,
    viewpoint_dir: np.ndarray,
    center: np.ndarray,
    image_size: int,
    target_occupancy: float = 0.5,
    min_dist: float = 0.8,
) -> float:
    """
    优化视角距离，使目标区域在画面中占据合适比例。

    参数:
        region_points: 目标区域点云 (N, 3)
        viewpoint_dir: 视角方向（单位向量）
        center: 观察中心点
        image_size: 图像尺寸
        target_occupancy: 目标占比
        min_dist: 最小距离阈值

    返回:
        优化后的距离
    """
    if region_points is None or len(region_points) == 0:
        return 1.5

    # 剔除离群点
    dists_to_center = np.linalg.norm(region_points - center, axis=1)
    limit_dist = np.percentile(dists_to_center, 99.0)
    core_mask = dists_to_center <= limit_dist
    core_points = region_points[core_mask]

    if len(core_points) == 0:
        return 1.5

    # 估算搜索范围
    max_radius = limit_dist
    fov = 60.0
    min_view_dist = max_radius / np.sin(np.radians(fov / 2.0))

    start_dist = max(min_dist, min_view_dist * 0.5)
    end_dist = max(start_dist * 3.0, min_view_dist * 3.0, 2.5)

    # 相机内参
    fx = image_size / (2.0 * np.tan(np.radians(fov) / 2.0))
    cam_params = {
        "intrinsic": {
            "width": image_size,
            "height": image_size,
            "fx": fx,
            "fy": fx,
            "cx": image_size / 2.0,
            "cy": image_size / 2.0,
            "fov": fov,
        }
    }
    intrinsic = create_camera_intrinsic_from_params(cam_params)

    test_dists = np.linspace(start_dist, end_dist, 20)
    best_dist = end_dist
    best_score = -float("inf")

    for dist in test_dists:
        eye = center + viewpoint_dir * dist
        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=center)

        pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
            core_points, intrinsic, extrinsic, image_size=image_size
        )

        # 完整性检查
        completeness = np.sum(valid_mask) / len(core_points)
        if completeness < 0.95:
            continue

        valid_pixels = pixel_coords[valid_mask]
        if len(valid_pixels) == 0:
            continue

        # 计算占比
        min_xy = np.min(valid_pixels, axis=0)
        max_xy = np.max(valid_pixels, axis=0)
        w = max_xy[0] - min_xy[0]
        h = max_xy[1] - min_xy[1]
        occupancy = (w * h) / (image_size * image_size)

        # 评分
        score = 1.0 - abs(occupancy - target_occupancy)

        # 边缘惩罚
        margin = image_size * 0.05
        if min_xy[0] < margin or min_xy[1] < margin or max_xy[0] > image_size - margin or max_xy[1] > image_size - margin:
            score -= 0.2

        if score > best_score:
            best_score = score
            best_dist = dist

    return best_dist


def select_best_viewpoints(
    region_points: np.ndarray,
    center: np.ndarray,
    image_size: int,
    num_views: int = 4,
    view_radius: float = 1.5,
) -> List[Tuple[np.ndarray, float]]:
    """
    从多个候选视角中选择最佳视角。

    参数:
        region_points: 目标区域点云 (N, 3)
        center: 观察中心点
        image_size: 图像尺寸
        num_views: 需要的视角数量
        view_radius: 基础视角半径

    返回:
        [(viewpoint, distance), ...] 列表
    """
    if region_points is None or len(region_points) == 0:
        # 返回默认视角
        default_views = [
            get_viewpoint_from_angles(0, 15, view_radius),
            get_viewpoint_from_angles(90, 15, view_radius),
            get_viewpoint_from_angles(180, 15, view_radius),
            get_viewpoint_from_angles(45, 45, view_radius),
        ]
        return [(v, view_radius) for v in default_views[:num_views]]

    # 采样候选视角
    candidate_points = sample_view_points(radius=1.0, partition=2)

    # 相机内参
    fov = 60.0
    fx = image_size / (2.0 * np.tan(np.radians(fov) / 2.0))
    cam_params = {
        "intrinsic": {
            "width": image_size,
            "height": image_size,
            "fx": fx,
            "fy": fx,
            "cx": image_size / 2.0,
            "cy": image_size / 2.0,
            "fov": fov,
        }
    }
    intrinsic = create_camera_intrinsic_from_params(cam_params)

    candidates = []
    for vp in candidate_points:
        direction = vp / np.linalg.norm(vp)

        # 优化距离
        dist = optimize_view_distance(region_points, direction, center, image_size, target_occupancy=0.5)
        eye = center + direction * dist

        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=center)
        pixel_coords, depths, valid_mask = project_points_to_image_with_depth(
            region_points, intrinsic, extrinsic, image_size=image_size
        )

        # 计算可见性
        visibility = np.sum(valid_mask) / len(region_points) if len(region_points) > 0 else 0

        if visibility < 0.3:
            continue

        # 计算占比
        valid_pixels = pixel_coords[valid_mask]
        if len(valid_pixels) > 0:
            min_xy = np.min(valid_pixels, axis=0)
            max_xy = np.max(valid_pixels, axis=0)
            w = max_xy[0] - min_xy[0]
            h = max_xy[1] - min_xy[1]
            occupancy = (w * h) / (image_size * image_size)
        else:
            occupancy = 0

        # 评分
        score = visibility * 5.0 + (1.0 - abs(occupancy - 0.5)) * 3.0

        candidates.append({
            "direction": direction,
            "dist": dist,
            "eye": eye,
            "score": score,
            "visibility": visibility,
        })

    # 按分数排序
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 选择互相独立的视角
    selected = []
    DISTINCTNESS_THRESHOLD = 0.6

    for cand in candidates:
        if len(selected) >= num_views:
            break

        is_distinct = True
        for sel in selected:
            if np.dot(cand["direction"], sel["direction"]) > DISTINCTNESS_THRESHOLD:
                is_distinct = False
                break

        if is_distinct:
            selected.append(cand)

    # 如果选择的不够，用默认视角补充
    if len(selected) < num_views:
        default_angles = [(0, 15), (90, 15), (180, 15), (45, 45), (270, 15), (135, 30)]
        for az, el in default_angles:
            if len(selected) >= num_views:
                break
            vp = get_viewpoint_from_angles(az, el, view_radius)
            direction = vp / np.linalg.norm(vp)

            is_distinct = True
            for sel in selected:
                if np.dot(direction, sel["direction"]) > DISTINCTNESS_THRESHOLD:
                    is_distinct = False
                    break

            if is_distinct:
                dist = optimize_view_distance(region_points, direction, center, image_size)
                selected.append({"direction": direction, "dist": dist, "eye": center + direction * dist})

    return [(s["eye"], s["dist"]) for s in selected]


def render_siblings_multiview(
    renderer: Open3DRenderer,
    viewpoints: List[Tuple[np.ndarray, float]],
    image_size: int,
    all_points: np.ndarray,
    sibling_indices_list: List[np.ndarray],
    sibling_colors: List[Tuple[int, int, int]],
    model: Optional[Any] = None,
) -> List[np.ndarray]:
    """
    返回若干张拼接图：左 clean（GLB渲染），右 siblings highlight（点云渲染，所有兄弟节点用不同颜色）。

    参数:
        renderer: GLB 渲染器
        viewpoints: 视角列表，每个元素是 (viewpoint, distance)
        image_size: 图像尺寸
        all_points: 完整点云 (N, 3)
        sibling_indices_list: 每个兄弟节点的点索引列表
        sibling_colors: 每个兄弟节点的颜色列表
        model: GLB 模型对象（用于在清空场景后重新加载）
    """
    center = np.array([0, 0, 0], dtype=np.float32)

    need_pc_render = (
        OPEN3D_AVAILABLE 
        and all_points is not None 
        and len(all_points) > 0 
        and sibling_indices_list
    )

    # 阶段 1：先用主渲染器渲染所有 clean 图像
    clean_images: List[np.ndarray] = []
    eye_list: List[np.ndarray] = []
    
    for vp_item in viewpoints:
        if isinstance(vp_item, tuple) and len(vp_item) == 2:
            eye, dist = vp_item
        else:
            eye = vp_item
        eye_list.append(eye)
        
        clean, _ = renderer.render_view(eye, center=center, return_depth=True)
        clean_images.append(clean)

    out: List[np.ndarray] = []
    
    if not need_pc_render:
        # 不需要点云渲染，直接返回 clean + clean
        for clean in clean_images:
            combined = np.concatenate([clean, clean], axis=1)
            out.append(combined)
        return out

    # 阶段 2：完全销毁主渲染器，释放 OpenGL 上下文
    renderer.cleanup()
    gc.collect()  # 强制垃圾回收，确保 OpenGL 资源被释放

    # 阶段 3：创建点云渲染器，渲染所有高亮图像
    pc_renderer = rendering.OffscreenRenderer(image_size, image_size)
    
    for i, (clean, eye) in enumerate(zip(clean_images, eye_list)):
        highlighted = render_pointcloud_siblings(
            all_points=all_points,
            sibling_indices_list=sibling_indices_list,
            sibling_colors=sibling_colors,
            viewpoint=eye,
            center=center,
            image_size=image_size,
            point_size=3.0,
            dim_factor=0.3,
            pc_renderer=pc_renderer,
        )
        if highlighted is None:
            highlighted = clean.copy()

        combined = np.concatenate([clean, highlighted], axis=1)
        out.append(combined)

    # 阶段 4：销毁点云渲染器
    del pc_renderer
    gc.collect()  # 强制垃圾回收，确保 OpenGL 资源被释放

    # 阶段 5：重新创建主渲染器并加载模型（供后续调用使用）
    if model is not None:
        renderer.setup()
        renderer.upload_model(model)

    return out


def _collect_children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = node.get("children")
    if isinstance(ch, list):
        return [x for x in ch if isinstance(x, dict)]
    return []


def _node_primary_name_str(node: Dict[str, Any]) -> str:
    """
    给 prompt / log 用的“主名称”字符串：
    - final_name/_final_name 优先
    - 否则取候选名列表的第一个（兼容 name 为 list）
    """
    if not isinstance(node, dict):
        return ""
    fn = node.get("final_name") or node.get("_final_name")
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    cands = _as_name_candidates(node)
    return cands[0] if cands else ""


@dataclass
class NodeResult:
    node_id: str
    level: int
    parent_id: Optional[str]
    gt_name: str
    final_name: str
    caption: str
    caption_json: Dict[str, Any]


@dataclass
class QAItem:
    qa_type: str
    question: str
    think: str
    answer: str
    brief: str
    object_id: str
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    extra: Dict[str, Any] = None


def _apply_transform_to_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """对 Nx3 点集应用 4x4 齐次变换（Open3D normalize_model 返回的 transform）。"""
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        return points_xyz
    pts = points_xyz[:, :3].astype(np.float64, copy=False)
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    out = (transform_4x4 @ homo.T).T[:, :3]
    return out.astype(np.float32)


def _partnext_hierarchy_to_tree(hierarchy: Any, object_id: str) -> Dict[str, Any]:
    """
    将 PartNeXt 的 hierarchyList（list 或 dict）转为内部 tree dict。
    字段对齐 visualize_pointcloud_gradio.py：
      - name
      - nodeId
      - maskId（叶子部件）
      - children
    """

    def convert_node(n: Dict[str, Any]) -> Dict[str, Any]:
        name = n.get("name", "Unknown")
        node_id = n.get("nodeId", None)
        mask_id = n.get("maskId", None)
        children = n.get("children", []) if isinstance(n.get("children", []), list) else []
        out = {
            "id": str(node_id) if node_id is not None else None,
            "name": name,
            "maskId": mask_id,
            "children": [convert_node(c) for c in children if isinstance(c, dict)],
        }
        return out

    if isinstance(hierarchy, list):
        roots = [convert_node(x) for x in hierarchy if isinstance(x, dict)]
        # PartNeXt_data 中 hierarchyList 典型格式是 list 且 len==1：该元素就是“真实树根”（nodeId 常为 0）
        if len(roots) == 1:
            r = roots[0]
            # 补齐 name（极少数情况下 name 为空）
            if not r.get("name"):
                r["name"] = object_id
            return r
        # 极少数情况下如果确实是 forest，则包一层虚拟 root 以保持单棵树输出
        return {"id": "root", "name": object_id, "children": roots, "virtual_root": True}
    if isinstance(hierarchy, dict):
        r = convert_node(hierarchy)
        if not r.get("id"):
            r["id"] = "root"
        if not r.get("name"):
            r["name"] = object_id
        return r
    raise ValueError(f"无法识别 PartNeXt hierarchy 类型: {type(hierarchy)}")


def _compute_region_points_bottom_up_partnext(
    node: Dict[str, Any],
    part_points_cache: Dict[str, np.ndarray],
    part_indices_cache: Optional[Dict[str, np.ndarray]] = None,
    max_points_per_node: int = 20000,
) -> np.ndarray:
    """给每个节点写入 _computed_region_points 和 _computed_region_indices（优先 maskId，否则用子节点 union）。"""
    children = _collect_children(node)
    for c in children:
        _compute_region_points_bottom_up_partnext(c, part_points_cache, part_indices_cache, max_points_per_node=max_points_per_node)

    mask_id = node.get("maskId", None)
    if mask_id is not None:
        pts = part_points_cache.get(str(mask_id), np.zeros((0, 3), dtype=np.float32))
        indices = part_indices_cache.get(str(mask_id), np.array([], dtype=np.int64)) if part_indices_cache else np.array([], dtype=np.int64)
    else:
        pts_list = [c.get("_computed_region_points") for c in children]
        pts_list = [p for p in pts_list if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] >= 3 and len(p) > 0]
        pts = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 3), dtype=np.float32)

        indices_list = [c.get("_computed_region_indices") for c in children]
        indices_list = [idx for idx in indices_list if isinstance(idx, np.ndarray) and len(idx) > 0]
        indices = np.concatenate(indices_list, axis=0) if indices_list else np.array([], dtype=np.int64)

    if len(pts) > max_points_per_node:
        sel = np.random.choice(len(pts), size=max_points_per_node, replace=False)
        pts = pts[sel]
        if len(indices) > 0:
            indices = indices[sel]

    node["_computed_region_points"] = pts.astype(np.float32, copy=False)
    node["_computed_region_indices"] = indices.astype(np.int64, copy=False) if len(indices) > 0 else np.array([], dtype=np.int64)
    return node["_computed_region_points"]


def _collect_descendant_mask_ids_partnext(node: Dict[str, Any]) -> List[str]:
    """
    收集某个节点子树下所有叶子 maskId（包含自身 if maskId 存在）。

    说明：
    - siblings 点云渲染阶段需要“完整覆盖”的索引集合，否则会出现本应属于某兄弟节点的点被当成灰色背景的问题。
    - `_compute_region_points_bottom_up_partnext()` 会对 `_computed_region_indices` 做点数上限裁剪（用于控内存/控渲染），
      因此 siblings 渲染不应直接依赖被裁剪后的 `_computed_region_indices`。
    """
    out: List[str] = []
    mid = node.get("maskId", None)
    if mid is not None:
        out.append(str(mid))
        return out
    for c in _collect_children(node):
        out.extend(_collect_descendant_mask_ids_partnext(c))
    return out


def _parse_name_caption_plain(text: str, fallback_name: str) -> Tuple[str, str]:
    """
    解析：
      Name: xxx
      Caption: yyy
    """
    name = ""
    caption = ""
    if not text:
        return fallback_name, ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("name:"):
            name = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("caption:"):
            caption = ln.split(":", 1)[1].strip()
    if not name:
        name = fallback_name
    if not caption:
        # 兜底：去掉可能的 Name 行，剩余作为 caption
        caption = "\n".join([ln for ln in lines if not ln.lower().startswith("name:")]).strip()
    return name, caption


# ============================================================================
# MLLM Prompt 模板（可单独优化）
# ============================================================================

SIBLING_CAPTION_PROMPT_TEMPLATE = """You are an expert annotator for 3D object parts. Please name and describe multiple highlighted sibling parts in the images.

## Context
- Object category/name: {global_name}
- Parent part name: {parent_name}

## About the images
- In each image, the **left half is the clean render** (use it to describe the real appearance of the part), and the **right half is a point-cloud highlight overlay** (ONLY for locating the part).
- In the right half, different colors indicate different sibling parts; gray indicates other regions.
- You are given {num_views} images from different viewpoints:
{view_descriptions}

## Sibling parts to annotate
{sibling_descriptions}

## Naming guidelines
- The "Name candidates" provided above are preliminary labels that may be inaccurate—they could be too vague, too broad, too narrow, or simply wrong.
- **Your task is to determine the most accurate name based on what you actually see in the images.** The visual appearance is the ground truth.
- If the candidate name is reasonable, you may keep it; if not, provide a better one that precisely describes the highlighted region.

## Output format
For each part, output the following format (strict order, one block per part):

[Part 1]
Name: <the most accurate part name based on visual appearance, in English>
Caption: <2-4 sentences describing the 3D part itself>

[Part 2]
Name: <the most accurate part name based on visual appearance, in English>
Caption: <2-4 sentences>

... and so on, total {num_siblings} parts.

## Critical constraints
- The Caption must be based **strictly on the left clean renders** and describe the real properties of the 3D part itself.
- The description should cover visual appearance, geometric shape, material/finish, structural details, and plausible function.
- Each Caption must be **standalone** and **self-contained**: do NOT refer to any other part or to previous captions. Avoid cross-reference language such as "the first/second", "former/latter", "this/that one", "mirrors", "its sibling", or "as above". Prefer explicit noun phrases that re-identify the part within the caption.
- **Do NOT mention**:
  - anything about the right highlight images (colors, point clouds, highlighted regions, etc.)
  - any viewpoint/observation phrasing (e.g., "from this view", "we can see", "in the image", etc.)
- Write as if describing a real physical part directly, without referencing how it was observed.
"""

# 根节点/整物体：仅基于 clean 多视角渲染图生成最合适的全局名称与 caption（不使用点云/部件树）
OBJECT_CAPTION_PROMPT_TEMPLATE = """You are an expert annotator for 3D object understanding.

You are given {num_views} clean renders of the SAME 3D object from different viewpoints.

Task:
- Determine the most appropriate object name/category based strictly on the visual appearance.
- Write a detailed but concise caption that describes the object as a whole.

## Critical constraints
- The visual appearance is the ground truth.
- Do NOT mention images, viewpoints, rendering, point clouds, highlights, markers, dots, digits, part trees, captions, or any external sources.
- If uncertain, choose a more general but still meaningful object category name rather than inventing specifics.

## Output format (strict)
Name: <object name/category in English>
Caption: <2-5 sentences describing the object as a whole>
"""

# 视角描述模板
VIEW_ANGLE_DESCRIPTIONS = [
    "Front view (azimuth 0°, elevation 15°)",
    "Right view (azimuth 90°, elevation 15°)",
    "Back view (azimuth 180°, elevation 15°)",
    "Left view (azimuth 270°, elevation 15°)",
    "Front-right top view (azimuth 45°, elevation 45°)",
    "Back-left top view (azimuth 135°, elevation 45°)",
]


# ============================================================================
# QA Prompt 模板（用于训练 QA/CoT 数据生成）
# ============================================================================

OBJECT_QA_PROMPT_TEMPLATE = """You are an annotation system for generating training data for 3D object understanding. You will receive:
1) Multi-view rendered images (global views; use them as primary evidence)
2) A part tree with part names and captions (auxiliary context; use it to improve global consistency and coverage, but do NOT treat it as direct visual evidence)

Generate ONE QA sample answering: "What is this 3D object?"

## Critical constraints (Think & Answer)
- Think must be 4-8 sentences in English. You may use all inputs to infer details, BUT write the Think as if you are directly inspecting ONLY the 3D object itself, with no access to images, part trees, captions, or any other external information. Do not mention any sources or how you know.
- Do NOT mention viewpoints, images, the part tree, captions, observation phrasing, point clouds, highlighted regions, or highlight colors.
- Answer must be in English: first give a clear object category/name, then add ONE short intro sentence (≤ 25 words).
- IMPORTANT: The part tree may contain errors. If the part tree conflicts with what you can infer from the object's visual appearance, prioritize the visual appearance.

## Input (part tree summary)
{tree_summary}

## Output format (strict JSON)
Output ONLY one JSON object and nothing else:
{{
  "qas": [
    {{
      "qa_type": "object_identity",
      "question": "...",
      "think": "...",
      "answer": "..."
    }}
  ]
}}
"""


PART_QA_PROMPT_TEMPLATE = """You are an annotation system for generating training data for 3D object understanding. You will receive:
1) Multi-view images (each image: left = clean render, right = highlighted target part for localization only)
2) A part tree with part names and captions (auxiliary context; use it to maintain consistency with the whole object, but do NOT treat it as direct visual evidence)
3) Current target part metadata (node_id / name / caption / parent-child relations)

Generate ONE QA sample answering: "What is this part?"

## Critical constraints (Think & Answer)
- Think must be 4-8 sentences in English. You may use all inputs to infer details, BUT write the Think as if you are directly inspecting ONLY the 3D object and this part itself, with no access to images, highlights, part trees, captions, node ids, or any other external information. Do not mention any sources or how you know.
- Do NOT mention viewpoints, images, the part tree, captions, node ids, observation phrasing, point clouds, highlighted regions, or highlight colors.
- Answer must be in English: first give a clear part category/name, then add ONE short intro sentence (≤ 25 words).
- IMPORTANT: The part tree may contain errors. If the part tree conflicts with what you can infer from the part's visual appearance, prioritize the visual appearance.

## Global part tree summary
{tree_summary}

## Target part
- node_id: {node_id}
- name: {node_name}
- caption: {node_caption}
- parent: {parent_name}
- children: {child_names}

## Output format (strict JSON)
Output ONLY one JSON object and nothing else:
{{
  "qas": [
    {{
      "qa_type": "part_identity",
      "question": "...",
      "think": "...",
      "answer": "..."
    }}
  ]
}}
"""


OTHER_QA_PROMPT_TEMPLATE = """You are an annotation system for generating training data for 3D object understanding. You will receive:
1) Multi-view rendered images (global views; use them as primary evidence)
2) A part tree summary (auxiliary context; use it to improve global consistency and completeness, but do NOT treat it as direct visual evidence)

Goal:
Generate an appropriate number of "other" QA samples (0 to {max_qas}) that complement the identity questions ("what is the object" / "what is the part") and improve 3D understanding.
Choose fewer questions for simple objects, and more questions for complex objects. Do NOT force the maximum.
Do NOT duplicate the identity questions.

## Key framing (Teacher → Student)
You have TWO internal roles:
- Teacher (you): You can use ALL inputs (images + part tree summary) to decide what to ask and what the correct answer should be.
- Student (the output): You must write the Think/Answer as if a student is directly inspecting ONLY the 3D object itself (no access to images, part trees, captions, node ids, renders, or any external sources).

The output fields "think" and "answer" MUST be written in the Student voice, while still being correct.

## Critical constraints (Question) — IMPORTANT
- Each question must be clear, concise, and written in English (not overly conversational).
- Keep questions minimally leading: do NOT embed the answer, do NOT provide extra hints, examples, or part lists in the question.
- Questions should focus on understanding the 3D object itself: overall geometry, functional affordances, and relationships among parts that are supported by the input summary.
- Do NOT ask anything that goes beyond the provided part tree summary (names and captions). Avoid brand, manufacturing, precise dimensions, hidden mechanisms, or extra parts not supported by the summary.
- Do NOT ask about material, color, texture, or decoration unless explicitly described in the summary.
- Do NOT mention the part tree, captions, node ids, rendering, or any observation/viewpoint phrasing.
- Avoid directional terms like "left/right/front/back/top/bottom".

## Critical constraints (Think & Answer) — Student voice ONLY
- Each QA must include Question/Think/Answer.
- Think must be 3-6 sentences in English, written as if directly inspecting ONLY the 3D object itself.
- Think MUST NOT mention or imply ANY sources (no: images, views, renders, part tree, summary, captions, node ids, prompts, or "the text says").
- Avoid phrases like: "from the images", "from the views", "the part tree", "the summary", "the caption", "it is described as", "the input states", "confirms".
- Answer must be in English: give a direct answer, then add ONE short intro sentence (≤ 25 words).
- IMPORTANT: The part tree may contain errors. If the tree conflicts with what the object visually looks like, prioritize the object's visual appearance.
- Avoid using double quotes inside Question/Think/Answer strings; rephrase if needed to keep strict JSON valid.

## Input (part tree summary)
{tree_summary}

## Output format (strict JSON)
Output ONLY one JSON object and nothing else:
{{
  "qas": [
    {{
      "qa_type": "other",
      "question": "...",
      "think": "...",
      "answer": "..."
    }}
  ]
}}
"""


OTHER_QA_PROMPT_TEMPLATE_FALLBACK = """You are an annotation system for generating training data for 3D object understanding.
You will receive:
1) Multi-view rendered images (global views; use them as primary evidence)
2) A part tree summary (auxiliary context; use it to keep consistency, but do NOT treat it as direct visual evidence)

Task:
- Generate 1 to {max_qas} "other" QA samples that complement identity questions.
- If the object is very simple, generate fewer questions (even 1). Prefer validity over quantity.

Teacher → Student framing (mandatory):
- You (Teacher) may use all inputs to craft good questions and determine the correct answers.
- The output "think" and "answer" must be written as if by a Student who can only inspect the 3D object itself.
- The Student must NEVER mention sources (images/views/rendering/part tree/summary/captions/node ids/input text).

Hard requirements:
- Output MUST be a single strict JSON object and nothing else.
- Questions MUST be answerable without going beyond the provided summary. Do NOT assume extra parts or mechanisms.
- Focus on the object itself (shape/affordance/part relations supported by the summary).
- Avoid asking about brand, manufacturing, precise dimensions, or hidden internals.
- Avoid asking about material/color/texture unless explicitly described in the summary.
- Do NOT use double quotes inside strings.

Think & Answer constraints (Student voice):
- Think: 3-6 sentences, English, written as if directly inspecting only the object.
- Think must not include phrases like "from the images/views", "the part tree/summary says", "described as", "the input states", etc.
- Answer: English, direct answer + one short intro sentence (≤ 25 words).

Input (part tree summary)
{tree_summary}

Output (strict JSON only)
{{
  "qas": [
    {{
      "qa_type": "other",
      "question": "...",
      "think": "...",
      "answer": "..."
    }}
  ]
}}
"""

PART_COUNT_QA_PROMPT_TEMPLATE = """You are an annotation system for generating training data for 3D object understanding.
You will receive a part tree summary with part names and captions (auxiliary context).

Task:
- If it is meaningful and unambiguous to ask part-counting questions from the given part tree summary, generate an appropriate number of part-counting QA samples (0 to {max_qas}).
- If it is NOT suitable (e.g., the tree summary is too incomplete/ambiguous, or counting would be misleading), output an empty list of qas.
- Choose fewer questions for simple objects, and more questions for complex objects. Do NOT force the maximum.

## Requirements
- Do NOT mention the part tree, captions, node ids, rendering, or any observation/viewpoint phrasing.
- Avoid directional terms like "left/right/front/back/top/bottom".
- The question should be about counts (e.g., number of legs, wheels, spokes, blades, drawers, etc.), but must NOT assume a specific part exists unless it is clearly supported by the tree summary.
- Think must be 3-6 sentences in English. You may use the tree summary to infer details, BUT write the Think as if you are directly inspecting ONLY the 3D object itself, with no access to part trees, captions, or any other external information. Do not mention any sources or how you know.
- Answer must be in English and include the final count explicitly.
- Avoid using double quotes inside Question/Think/Answer strings; rephrase if needed to keep strict JSON valid.

## Input (part tree summary)
{tree_summary}

## Output format (strict JSON)
Output ONLY one JSON object and nothing else:
{{
  "qas": [
    {{
      "qa_type": "part_count",
      "question": "...",
      "think": "...",
      "answer": "..."
    }}
  ]
}}
"""


def _summarize_tree_for_qa(root: Dict[str, Any]) -> str:
    """
    将整棵树（不压缩、不截断）展开为给 MLLM 的可读文本。

    注意：这里使用 DFS（先序）而不是 BFS。
    BFS 虽然 parent 字段是对的，但输出顺序会让缩进在视觉上“像是挂在上一行节点下面”，容易误读父子关系。
    """
    items: List[str] = []

    def rec(n: Dict[str, Any], lvl: int, pid: Optional[str]) -> None:
        nid = str(n.get("id", ""))
        name = _node_primary_name_str(n)
        cap = str(n.get("caption") or "")
        children = _collect_children(n)
        child_names: List[str] = []
        for c in children:
            cn = _node_primary_name_str(c)
            if cn:
                child_names.append(cn)
        prefix = "  " * lvl
        items.append(
            f"{prefix}- id={nid} parent={pid or 'None'} name={name} children={json.dumps(child_names, ensure_ascii=False)} caption={cap}"
        )
        for c in children:
            rec(c, lvl + 1, nid)

    rec(root, 0, None)
    return "\n".join(items)


def _collect_nodes_with_region_indices(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """收集包含 _computed_region_indices 的节点，用于后续单部件高亮渲染。"""
    out: List[Dict[str, Any]] = []
    q: List[Dict[str, Any]] = [root]
    while q:
        n = q.pop(0)
        idx = n.get("_computed_region_indices")
        if isinstance(idx, np.ndarray) and len(idx) > 0:
            out.append(n)
        for c in _collect_children(n):
            q.append(c)
    return out


def _build_parent_child_name_maps(root: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """返回: node_id -> parent_name, node_id -> child_names（均取 final_name 优先）。"""
    parent_name_map: Dict[str, str] = {}
    child_names_map: Dict[str, List[str]] = {}
    q: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = [(root, None)]
    while q:
        n, p = q.pop(0)
        nid = str(n.get("id"))
        p_name = ""
        if p is not None:
            p_name = _node_primary_name_str(p)
        parent_name_map[nid] = p_name
        children = _collect_children(n)
        cnames: List[str] = []
        for c in children:
            cn = _node_primary_name_str(c)
            if cn:
                cnames.append(cn)
            q.append((c, n))
        child_names_map[nid] = cnames
    return parent_name_map, child_names_map


def _downsample_full_pointcloud_and_remap_indices(
    full_points: np.ndarray,
    nodes: List[Dict[str, Any]],
    max_points: int,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    若 full_points 过大，对其下采样，并将每个节点的 _computed_region_indices 映射到新点云索引。
    返回: (new_points, node_id -> new_indices)
    """
    if full_points is None or len(full_points) == 0:
        return full_points, {}
    n = int(len(full_points))
    if max_points <= 0 or n <= max_points:
        remap: Dict[str, np.ndarray] = {}
        for nd in nodes:
            nid = str(nd.get("id"))
            idx = nd.get("_computed_region_indices")
            if isinstance(idx, np.ndarray) and len(idx) > 0:
                remap[nid] = idx.astype(np.int64, copy=False)
        return full_points, remap

    rng = np.random.RandomState(seed)
    sel = rng.choice(n, size=int(max_points), replace=False)
    sel = np.sort(sel).astype(np.int64)
    inv = np.full((n,), -1, dtype=np.int64)
    inv[sel] = np.arange(len(sel), dtype=np.int64)

    new_points = full_points[sel].astype(np.float32, copy=False)
    remap: Dict[str, np.ndarray] = {}
    for nd in nodes:
        nid = str(nd.get("id"))
        idx = nd.get("_computed_region_indices")
        if not isinstance(idx, np.ndarray) or len(idx) == 0:
            continue
        new_idx = inv[idx.astype(np.int64, copy=False)]
        new_idx = new_idx[new_idx >= 0]
        if len(new_idx) > 0:
            remap[nid] = new_idx
    return new_points, remap


def _render_clean_multiview(renderer: Open3DRenderer, viewpoints: List[Tuple[np.ndarray, float]], center: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    clean_images: List[np.ndarray] = []
    eyes: List[np.ndarray] = []
    for vp_item in viewpoints:
        eye = vp_item[0] if isinstance(vp_item, tuple) and len(vp_item) == 2 else vp_item
        eyes.append(np.asarray(eye, dtype=np.float32))
        img, _ = renderer.render_view(eye, center=center, return_depth=True)
        clean_images.append(img)
    return clean_images, eyes


def _render_single_part_highlight_multiview(
    pc_renderer: "rendering.OffscreenRenderer",
    all_points: np.ndarray,
    part_indices: np.ndarray,
    eyes: List[np.ndarray],
    clean_images: List[np.ndarray],
    center: np.ndarray,
    image_size: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for clean, eye in zip(clean_images, eyes):
        highlighted = render_pointcloud_highlight(
            all_points=all_points,
            highlight_indices=part_indices,
            viewpoint=eye,
            center=center,
            image_size=image_size,
            point_size=3.0,
            highlight_color=(255, 0, 0),
            dim_factor=0.3,
            pc_renderer=pc_renderer,
        )
        if highlighted is None:
            highlighted = clean.copy()
        out.append(np.concatenate([clean, highlighted], axis=1))
    return out


def generate_partnext_qa_dataset(
    client: MLLMClient,
    object_id: str,
    tree: Dict[str, Any],
    glb_path: str,
    image_size: int,
    viewpoints: List[Tuple[np.ndarray, float]],
    full_pointcloud: np.ndarray,
    qa_part_nodes: str = "leaf",
    qa_max_parts: int = 20,
    qa_num_other: int = 6,
    qa_enable_part_count: bool = True,
    qa_num_part_count: int = 3,
    qa_max_points: int = 200000,
    qa_max_tokens: int = 2048,
    qa_temperature: float = 0.3,
    seed: int = 0,
    qa_debug_print_prompts: bool = True,
    qa_debug_print_responses: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    生成 QA 数据集（jsonl 友好）。
    - 物体 QA：使用 clean 多视角图
    - 其他 QA：使用 clean 多视角图（数量 0..qa_num_other，模型自适应）
    - 部件计数 QA（可选）：仅基于树摘要生成（模型可生成 0..qa_num_part_count 条；若不适合则输出空 qas）
    - 部件 QA：使用 (clean | 目标部件高亮) 拼接多视角图
    """
    center = np.array([0, 0, 0], dtype=np.float32)
    tree_summary = _summarize_tree_for_qa(tree)
    parent_name_map, child_names_map = _build_parent_child_name_maps(tree)

    # 收集可渲染节点（需要 region_indices）
    nodes = _collect_nodes_with_region_indices(tree)
    # 不要把“整物体根节点/root”当作一个部件来问（即便 qa_part_nodes=all）
    # 兼容 PartNeXt 常见 root id=0，以及 forest 情况下的 virtual_root（id="root"）
    root_id = str(tree.get("id")) if tree.get("id") is not None else None
    nodes = [
        n
        for n in nodes
        if not (
            n is tree
            or (root_id is not None and str(n.get("id")) == root_id)
            or bool(n.get("virtual_root"))
        )
    ]
    if qa_part_nodes == "leaf":
        nodes = [n for n in nodes if len(_collect_children(n)) == 0]
    if qa_max_parts > 0:
        nodes = nodes[:qa_max_parts]

    # 点云过大时下采样，并 remap indices
    pc_points, idx_map = _downsample_full_pointcloud_and_remap_indices(
        full_points=full_pointcloud,
        nodes=nodes,
        max_points=qa_max_points,
        seed=seed,
    )

    # 渲染 clean 多视角
    renderer = Open3DRenderer(width=image_size, height=image_size)
    renderer.setup()
    model = renderer.load_model(glb_path)
    model, _ = renderer.normalize_model(model)
    renderer.upload_model(model)
    clean_images, eyes = _render_clean_multiview(renderer, viewpoints, center=center)
    renderer.cleanup()
    del renderer
    gc.collect()

    qa_items: List[Dict[str, Any]] = []
    # raw: 存模型原始输出；prompts: 存每次调用使用的 prompt 便于复盘
    qa_debug: Dict[str, Any] = {"object_id": object_id, "raw": {}, "prompts": {}}

    # ========== 物体是什么 ==========
    obj_prompt = OBJECT_QA_PROMPT_TEMPLATE.format(tree_summary=tree_summary)
    obj_resp = _call_mllm_for_qa(
        client=client,
        prompt=obj_prompt,
        images=clean_images,
        max_tokens=qa_max_tokens,
        temperature=qa_temperature,
        debug_label=f"object_identity | object_id={object_id}",
        debug_print_prompt=qa_debug_print_prompts,
        debug_print_response=qa_debug_print_responses,
    )
    qa_debug["prompts"]["object_identity"] = obj_prompt
    qa_debug["raw"]["object_identity"] = obj_resp.get("raw", "")
    qa_items.extend(_normalize_qa_items(obj_resp, object_id=object_id, qa_type_fallback="object_identity"))

    # ========== 其他形式 ==========
    if qa_num_other and qa_num_other > 0:
        other_prompt = OTHER_QA_PROMPT_TEMPLATE.format(tree_summary=tree_summary, max_qas=int(qa_num_other))
        other_resp = _call_mllm_for_qa(
            client=client,
            prompt=other_prompt,
            images=clean_images,
            max_tokens=qa_max_tokens,
            temperature=max(0.4, qa_temperature),
            debug_label=f"other | object_id={object_id} | num_qas={int(qa_num_other)}",
            debug_print_prompt=qa_debug_print_prompts,
            debug_print_response=qa_debug_print_responses,
        )
        qa_debug["prompts"]["other"] = other_prompt
        qa_debug["raw"]["other"] = other_resp.get("raw", "")

        other_items = _normalize_qa_items(other_resp, object_id=object_id, qa_type_fallback="other")
        # 回退策略：当 other 解析结果为空时（包括模型返回合法但空的 {"qas": []}），重试 1 次。
        # 说明：按数据构造需求，这种“解析为空”通常意味着模型未按预期生成 other QA，需要用更强约束 prompt 再试一次。
        if not other_items:
            fallback_prompt = OTHER_QA_PROMPT_TEMPLATE_FALLBACK.format(
                tree_summary=tree_summary,
                max_qas=int(max(1, qa_num_other)),
            )
            other_resp2 = _call_mllm_for_qa(
                client=client,
                prompt=fallback_prompt,
                images=clean_images,
                max_tokens=qa_max_tokens,
                temperature=max(0.2, qa_temperature),
                debug_label=f"other_fallback | object_id={object_id} | num_qas={int(qa_num_other)}",
                debug_print_prompt=qa_debug_print_prompts,
                debug_print_response=qa_debug_print_responses,
            )
            qa_debug["prompts"]["other_fallback"] = fallback_prompt
            qa_debug["raw"]["other_fallback"] = other_resp2.get("raw", "")
            other_items = _normalize_qa_items(other_resp2, object_id=object_id, qa_type_fallback="other")

        qa_items.extend(other_items)

    # ========== 部件计数（可选，且可不生成） ==========
    if qa_enable_part_count and qa_num_part_count and qa_num_part_count > 0:
        # 简单启发式：至少有 2 个叶子部件才值得问“数量”类问题
        def _count_leaf_nodes(n: Dict[str, Any]) -> int:
            ch = _collect_children(n)
            if not ch:
                return 1
            return sum(_count_leaf_nodes(c) for c in ch)

        leaf_cnt = _count_leaf_nodes(tree)
        if leaf_cnt >= 2:
            count_prompt = PART_COUNT_QA_PROMPT_TEMPLATE.format(
                tree_summary=tree_summary,
                max_qas=int(qa_num_part_count),
            )
            count_resp = _call_mllm_for_qa(
                client=client,
                prompt=count_prompt,
                images=[],  # 仅基于部件树 caption
                max_tokens=min(1024, qa_max_tokens),
                temperature=qa_temperature,
                debug_label=f"part_count | object_id={object_id} | max_qas={int(qa_num_part_count)}",
                debug_print_prompt=qa_debug_print_prompts,
                debug_print_response=qa_debug_print_responses,
            )
            qa_debug["prompts"]["part_count"] = count_prompt
            qa_debug["raw"]["part_count"] = count_resp.get("raw", "")
            qa_items.extend(_normalize_qa_items(count_resp, object_id=object_id, qa_type_fallback="part_count"))

    # ========== 部件是什么 ==========
    if not OPEN3D_AVAILABLE or pc_points is None or len(pc_points) == 0:
        return qa_items, qa_debug

    pc_renderer = rendering.OffscreenRenderer(image_size, image_size)
    part_raw: Dict[str, str] = {}
    part_prompts: Dict[str, str] = {}

    for nd in nodes:
        nid = str(nd.get("id"))
        indices = idx_map.get(nid)
        if indices is None or len(indices) == 0:
            continue

        node_name = str(nd.get("final_name") or nd.get("_final_name") or nd.get("name") or "")
        node_caption = str(nd.get("caption") or "")
        parent_name = parent_name_map.get(nid, "")
        child_names = child_names_map.get(nid, [])

        part_images = _render_single_part_highlight_multiview(
            pc_renderer=pc_renderer,
            all_points=pc_points,
            part_indices=indices,
            eyes=eyes,
            clean_images=clean_images,
            center=center,
            image_size=image_size,
        )

        part_prompt = PART_QA_PROMPT_TEMPLATE.format(
            tree_summary=tree_summary,
            node_id=nid,
            node_name=node_name,
            node_caption=node_caption,
            parent_name=parent_name,
            child_names=json.dumps(child_names, ensure_ascii=False),
        )
        part_resp = _call_mllm_for_qa(
            client=client,
            prompt=part_prompt,
            images=part_images,
            max_tokens=qa_max_tokens,
            temperature=qa_temperature,
            debug_label=f"part_identity | object_id={object_id} | node_id={nid} | node_name={node_name}",
            debug_print_prompt=qa_debug_print_prompts,
            debug_print_response=qa_debug_print_responses,
        )
        part_prompts[nid] = part_prompt
        part_raw[nid] = part_resp.get("raw", "")
        qa_items.extend(_normalize_qa_items(part_resp, object_id=object_id, qa_type_fallback="part_identity", node=nd))

    del pc_renderer
    gc.collect()
    qa_debug["raw"]["part_identity"] = part_raw
    qa_debug["prompts"]["part_identity"] = part_prompts

    return qa_items, qa_debug


def _build_sibling_caption_prompt(
    global_name: str,
    parent_name: str,
    sibling_info_list: List[Dict[str, Any]],
    num_views: int,
) -> str:
    """
    根据模板构建 MLLM prompt。
    
    参数:
        global_name: 物体整体名称
        parent_name: 父部件名称
        sibling_info_list: 兄弟节点信息列表
        num_views: 视角数量
    
    返回:
        构建好的 prompt 字符串
    """
    # 构建视角描述
    view_lines = []
    for i in range(min(num_views, len(VIEW_ANGLE_DESCRIPTIONS))):
        view_lines.append(f"  - Image {i+1}: {VIEW_ANGLE_DESCRIPTIONS[i]}")
    view_descriptions = "\n".join(view_lines)
    
    # 构建兄弟节点描述
    sibling_lines = []
    for info in sibling_info_list:
        sibling_lines.append(f"\n### Part {info['index']} ({info['color']})")
        sibling_lines.append(f"- Name candidates: {json.dumps(info['name_candidates'], ensure_ascii=False)}")
        if info['child_names']:
            sibling_lines.append(f"- Child parts: {json.dumps(info['child_names'], ensure_ascii=False)}")
    sibling_descriptions = "\n".join(sibling_lines)
    
    return SIBLING_CAPTION_PROMPT_TEMPLATE.format(
        global_name=global_name or "Unknown",
        parent_name=parent_name or "Root",
        num_views=num_views,
        view_descriptions=view_descriptions,
        sibling_descriptions=sibling_descriptions,
        num_siblings=len(sibling_info_list),
    )


def name_and_caption_siblings(
    client: MLLMClient,
    global_name: str,
    parent_name: str,
    siblings: List[Dict[str, Any]],
    color_names: List[str],
    images: List[np.ndarray],
    view_indices: Optional[List[int]] = None,
    debug_save_dir: Optional[str] = None,
    debug_group_id: str = "",
) -> List[Dict[str, Any]]:
    """
    一次调用为所有兄弟节点生成：最终部件名 + 非结构化 caption。

    参数:
        siblings: 兄弟节点列表
        color_names: 每个兄弟对应的颜色名称列表
        images: 多视角图像列表
        view_indices: 视角索引列表（对应 VIEW_ANGLE_DESCRIPTIONS）
        debug_save_dir: 调试图片保存目录（可选）
        debug_group_id: 调试用的组标识

    返回:
        每个兄弟节点的结果列表 [{"name": ..., "caption": ..., "raw": ...}, ...]
    """
    # 收集每个兄弟的候选名称
    sibling_info_list = []
    for i, (sibling, color) in enumerate(zip(siblings, color_names)):
        name_cands = _as_name_candidates(sibling)
        child_nodes = _collect_children(sibling)
        child_names = [_as_name_candidates(c)[0] for c in child_nodes if _as_name_candidates(c)]
        sibling_info_list.append({
            "index": i + 1,
            "color": color,
            "name_candidates": name_cands,
            "child_names": child_names,
        })

    # 使用模板构建 prompt
    prompt = _build_sibling_caption_prompt(
        global_name=global_name,
        parent_name=parent_name,
        sibling_info_list=sibling_info_list,
        num_views=len(images),
    )

    # ========== 调试输出 ==========
    print("\n" + "=" * 80)
    print(f"[MLLM REQUEST] Group: {debug_group_id}, Siblings: {len(siblings)}")
    print("=" * 80)
    print("\n--- PROMPT START ---")
    print(prompt)
    print("--- PROMPT END ---\n")

    # 保存调试图片
    if debug_save_dir:
        debug_img_dir = os.path.join(debug_save_dir, "mllm_inputs")
        os.makedirs(debug_img_dir, exist_ok=True)
        for i, img in enumerate(images):
            img_path = os.path.join(debug_img_dir, f"{debug_group_id}_view{i}.png")
            Image.fromarray(img).save(img_path)
            print(f"[DEBUG] Saved input image: {img_path}")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        img_b64 = client.encode_image(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    print(f"[MLLM] Calling with {len(images)} images...")
    resp = client.call([{"role": "user", "content": content}], max_tokens=2048, enable_thinking=False, temperature=0.3)

    print("\n--- MLLM RESPONSE START ---")
    print(resp)
    print("--- MLLM RESPONSE END ---\n")

    # 解析多个部件的输出
    results = _parse_siblings_caption(resp, siblings, color_names)

    print(f"[PARSED] {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Name: {r.get('name')}, Caption: {r.get('caption')[:50]}..." if r.get('caption') else f"  [{i+1}] Name: {r.get('name')}, Caption: (empty)")

    return results


def _call_mllm_for_qa(
    client: MLLMClient,
    prompt: str,
    images: List[np.ndarray],
    max_tokens: int = 2048,
    temperature: float = 0.3,
    debug_label: str = "",
    debug_print_prompt: bool = False,
    debug_print_response: bool = False,
) -> Dict[str, Any]:
    """
    调用 MLLM 生成 QA，期望返回:
      {"qas": [ {qa_type, question, think, answer, brief, ...}, ... ]}
    """
    if debug_print_prompt:
        print("\n" + "=" * 80)
        print(
            f"[QA MLLM REQUEST] {debug_label or 'qa'} | images={len(images)} | max_tokens={max_tokens} | temperature={temperature}"
        )
        print("=" * 80)
        print("\n--- PROMPT START ---")
        print(prompt)
        print("--- PROMPT END ---\n")

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        img_b64 = client.encode_image(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    resp = client.call(
        [{"role": "user", "content": content}],
        max_tokens=max_tokens,
        enable_thinking=False,
        temperature=temperature,
    )

    if debug_print_response:
        print("\n" + "=" * 80)
        print(
            f"[QA MLLM RESPONSE] {debug_label or 'qa'} | images={len(images)} | max_tokens={max_tokens} | temperature={temperature}"
        )
        print("=" * 80)
        print("\n--- RESPONSE START ---")
        print(resp)
        print("--- RESPONSE END ---\n")

    parsed = _extract_first_json_value(resp)
    if isinstance(parsed, dict):
        out = {"raw": resp, "_parsed_ok": True, "_salvaged": False, **parsed}
        if debug_print_response:
            qas_cnt = out.get("qas")
            qas_cnt = len(qas_cnt) if isinstance(qas_cnt, list) else 0
            print(f"[QA PARSE] parsed_ok=True salvaged=False qas={qas_cnt}\n")
        return out

    # 解析失败（常见：输出被截断导致 JSON 不配平），尽量从 qas 数组里恢复已完成的对象
    salvaged_qas = _extract_qas_list_best_effort(resp)
    if salvaged_qas:
        out = {"raw": resp, "_parsed_ok": False, "_salvaged": True, "qas": salvaged_qas}
        if debug_print_response:
            print(f"[QA PARSE] parsed_ok=False salvaged=True qas={len(salvaged_qas)}\n")
        return out
    out = {"raw": resp, "_parsed_ok": False, "_salvaged": False, "qas": []}
    if debug_print_response:
        print("[QA PARSE] parsed_ok=False salvaged=False qas=0\n")
    return out


def _normalize_qa_items(
    obj: Dict[str, Any],
    object_id: str,
    qa_type_fallback: str,
    node: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    qas = obj.get("qas") if isinstance(obj, dict) else None
    if not isinstance(qas, list):
        return []
    out: List[Dict[str, Any]] = []
    for q in qas:
        if not isinstance(q, dict):
            continue
        item = QAItem(
            qa_type=str(q.get("qa_type") or qa_type_fallback),
            question=str(q.get("question") or "").strip(),
            # 兼容旧字段 cot：优先用 think，否则回退到 cot
            think=str(q.get("think") or q.get("cot") or "").strip(),
            answer=str(q.get("answer") or "").strip(),
            brief=str(q.get("brief") or "").strip(),
            object_id=object_id,
            node_id=str(node.get("id")) if node is not None and node.get("id") is not None else None,
            node_name=str(node.get("final_name") or node.get("name")) if node is not None else None,
            extra={k: v for k, v in q.items() if k not in {"qa_type", "question", "think", "cot", "answer", "brief"}},
        )
        out.append(asdict(item))
    return out


def _parse_siblings_caption(
    text: str,
    siblings: List[Dict[str, Any]],
    color_names: List[str],
) -> List[Dict[str, Any]]:
    """
    解析 MLLM 输出的多兄弟 caption。

    Expected format:
    [Part 1]
    Name: xxx
    Caption: yyy

    [Part 2]
    ...
    """
    import re

    results = []
    n = len(siblings)

    # Split by section headers: support both Chinese and English markers.
    pattern = r'\[(?:部件|Part)\s*(\d+)\]'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    # parts 格式: ['前导文本', '1', '部件1内容', '2', '部件2内容', ...]
    parsed = {}
    i = 1
    while i < len(parts) - 1:
        try:
            idx = int(parts[i])
            content = parts[i + 1]
            parsed[idx] = content
            i += 2
        except (ValueError, IndexError):
            i += 1

    for idx, sibling in enumerate(siblings):
        fallback_name = _as_name_candidates(sibling)
        fallback_name = fallback_name[0] if fallback_name else f"part_{sibling.get('id', 'unknown')}"

        content = parsed.get(idx + 1, "")
        name = fallback_name
        caption = ""

        if content:
            lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
            for ln in lines:
                if ln.lower().startswith("name:"):
                    name = ln.split(":", 1)[1].strip() or fallback_name
                elif ln.lower().startswith("caption:"):
                    caption = ln.split(":", 1)[1].strip()

            # 如果没有找到 Caption: 标签，把剩余内容作为 caption
            if not caption:
                non_name_lines = [ln for ln in lines if not ln.lower().startswith("name:")]
                caption = " ".join(non_name_lines)

        results.append({
            "name": name,
            "caption": caption,
            "color": color_names[idx],
            "raw": content,
        })

    return results


def build_bfs_list(root: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]]:
    q: List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]] = [(root, 0, None)]
    out: List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]] = []
    while q:
        n, lvl, p = q.pop(0)
        out.append((n, lvl, p))
        for c in _collect_children(n):
            q.append((c, lvl + 1, n))
    return out


def build_dfs_list(root: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]]:
    out: List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]] = []

    def rec(n: Dict[str, Any], lvl: int, p: Optional[Dict[str, Any]]):
        out.append((n, lvl, p))
        for c in _collect_children(n):
            rec(c, lvl + 1, n)

    rec(root, 0, None)
    return out


def process_partnext_captioning(
    glb_dir: str,
    ann_dir: str,
    object_id: str,
    output_dir: str,
    image_size: int,
    num_views: int,
    view_radius: float,
    traversal: str,
    save_images: bool,
    dry_run: bool,
    mllm_client: Optional[MLLMClient],
    part_sample_points: int = 5000,
    generate_qa: bool = False,
    qa_output_path: Optional[str] = None,
    qa_part_nodes: str = "leaf",
    qa_max_parts: int = 20,
    qa_num_other: int = 6,
    qa_enable_part_count: bool = True,
    qa_num_part_count: int = 3,
    qa_max_points: int = 200000,
    qa_max_tokens: int = 2048,
    qa_temperature: float = 0.3,
    seed: int = 0,
    qa_debug_print_prompts: bool = True,
    qa_debug_print_responses: bool = False,
) -> Dict[str, Any]:
    """
    直接读取 PartNeXt 的 hierarchyList（nodeId/maskId/name/children），并用 maskId 对应的叶子部件点集做高亮渲染。
    兼容你在 visualize_pointcloud_gradio.py 里展示的格式。
    """
    if not PARTNEXT_AVAILABLE:
        raise RuntimeError(
            "PartNeXt 不可用：请确保仓库内存在 `PartNeXt/PartNeXt_lib`，并且环境已安装依赖（datasets/trimesh 等）。"
        )

    dataset = PartNeXtDataset(glb_dir, ann_dir)
    pn_obj = dataset.load_object(object_id)
    if pn_obj is None:
        raise RuntimeError(f"无法加载 PartNeXt 对象: {object_id}")

    # 推导 glb_path（复用 dataset 索引字段 type_id/model_id）
    row = dataset.index.get(object_id)
    if row is None:
        raise RuntimeError(f"PartNeXt 标注里找不到 model_id={object_id}")
    glb_path = os.path.join(glb_dir, row["type_id"], row["model_id"] + ".glb")

    hierarchy = pn_obj.get_hierarchy()
    tree = _partnext_hierarchy_to_tree(hierarchy, object_id=object_id)
    # ========= 预处理：链压缩 + name 统一为 list =========
    # 后续所有操作（region 计算 / caption / QA）都基于预处理后的树
    preprocess_part_tree(tree, object_id=object_id)

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    if save_images:
        os.makedirs(images_dir, exist_ok=True)

    # Open3D 渲染器 + 归一化变换：对 PartNeXt 采样点应用同样变换，保证与渲染对齐
    renderer = Open3DRenderer(width=image_size, height=image_size)
    renderer.setup()
    model = renderer.load_model(glb_path)
    model, norm_T = renderer.normalize_model(model)
    renderer.upload_model(model)

    # 默认视角（会在遍历节点时根据区域动态优化）
    default_view_radius = view_radius

    # 预采样每个 maskId 的点集（trimesh），并应用 Open3D 的归一化 transform
    all_parts = pn_obj.get_all_parts()  # {part_id(str): trimesh}
    part_points_cache: Dict[str, np.ndarray] = {}
    part_indices_cache: Dict[str, np.ndarray] = {}  # 记录每个部件在完整点云中的索引
    all_points_list: List[np.ndarray] = []
    current_idx = 0

    for pid, mesh in all_parts.items():
        try:
            if hasattr(mesh, "sample"):
                pts = mesh.sample(int(part_sample_points))
            else:
                pts = np.asarray(mesh.vertices)
        except Exception:
            pts = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))))
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
            pts = _apply_transform_to_points(pts[:, :3], norm_T)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
        part_points_cache[str(pid)] = pts

        # 记录索引范围
        if len(pts) > 0:
            indices = np.arange(current_idx, current_idx + len(pts))
            part_indices_cache[str(pid)] = indices
            all_points_list.append(pts)
            current_idx += len(pts)
        else:
            part_indices_cache[str(pid)] = np.array([], dtype=np.int64)

    # 构建完整点云（用于点云渲染）
    if all_points_list:
        full_pointcloud = np.concatenate(all_points_list, axis=0).astype(np.float32)
    else:
        full_pointcloud = np.zeros((0, 3), dtype=np.float32)

    # 自底向上：为每个节点生成 region_points 和 region_indices
    _compute_region_points_bottom_up_partnext(tree, part_points_cache, part_indices_cache)
    # 你的流程：单子节点父名下沉到子节点候选名
    propagate_single_child_names_to_child_candidates(tree)

    # 遍历顺序
    if traversal == "bfs":
        order = build_bfs_list(tree)
    elif traversal == "dfs":
        order = build_dfs_list(tree)
    else:
        raise ValueError("--traversal 只能是 bfs 或 dfs")

    # root/global 名称：优先使用根节点候选名（兼容 name 为 list）
    global_name = _node_primary_name_str(tree) or object_id
    global_caption = ""
    node_caption_cache: Dict[str, Dict[str, Any]] = {}
    results: List[NodeResult] = []

    # 按兄弟组处理：先收集每个父节点的所有子节点
    # 构建 parent_id -> [children] 映射
    from collections import defaultdict
    parent_to_children: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)

    for node, lvl, parent in order:
        node_id = str(node.get("id") or f"level{lvl}_idx{len(parent_to_children)}")
        node["id"] = node_id
        parent_id = str(parent.get("id")) if isinstance(parent, dict) and parent.get("id") is not None else None
        parent_to_children[parent_id].append((node, lvl, parent))

    # 按兄弟组处理
    center = np.array([0, 0, 0], dtype=np.float32)

    # 固定的全局视角（所有节点使用相同视角，只是高亮的点云不同）
    fixed_views = [
        get_viewpoint_from_angles(0, 15, default_view_radius),
        get_viewpoint_from_angles(90, 15, default_view_radius),
        get_viewpoint_from_angles(180, 15, default_view_radius),
        get_viewpoint_from_angles(270, 15, default_view_radius),
        get_viewpoint_from_angles(45, 45, default_view_radius),
        get_viewpoint_from_angles(135, 45, default_view_radius),
    ]
    viewpoints = [(v, default_view_radius) for v in fixed_views[:num_views]]

    # ========== 根节点全局命名与 caption（仅 clean 多视角，不用点云/部件树）==========
    # 一切以视觉渲染图为准，允许部件树存在错误。
    root_id = str(tree.get("id") or "root")
    clean_root_images: List[np.ndarray] = []
    for vp_item in viewpoints:
        eye = vp_item[0] if isinstance(vp_item, tuple) and len(vp_item) == 2 else vp_item
        img, _ = renderer.render_view(eye, center=center, return_depth=True)
        clean_root_images.append(img)

    if save_images:
        debug_img_dir = os.path.join(output_dir, "mllm_inputs")
        os.makedirs(debug_img_dir, exist_ok=True)
        for i, img in enumerate(clean_root_images):
            Image.fromarray(img).save(os.path.join(debug_img_dir, f"ROOT_{root_id}_view{i}.png"))

    if (not dry_run) and (mllm_client is not None):
        obj_cap_prompt = OBJECT_CAPTION_PROMPT_TEMPLATE.format(num_views=len(clean_root_images))
        content: List[Dict[str, Any]] = [{"type": "text", "text": obj_cap_prompt}]
        for img in clean_root_images:
            img_b64 = mllm_client.encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        resp = mllm_client.call(
            [{"role": "user", "content": content}],
            max_tokens=1024,
            enable_thinking=False,
            temperature=0.3,
        )
        root_name, root_caption = _parse_name_caption_plain(resp, fallback_name=global_name)
        global_name = root_name
        global_caption = root_caption
        tree["_final_name"] = root_name
        tree["final_name"] = root_name
        tree["caption"] = root_caption
        node_caption_cache[root_id] = {"name": root_name, "caption": root_caption, "raw": resp}
    else:
        # dry_run：不调用 MLLM
        tree["_final_name"] = global_name
        tree["final_name"] = global_name
        tree["caption"] = ""

    for parent_id, siblings_data in parent_to_children.items():
        if not siblings_data:
            continue

        # 获取兄弟节点列表
        siblings = [s[0] for s in siblings_data]
        lvl = siblings_data[0][1]
        parent = siblings_data[0][2]

        # 根节点（parent=None）已经单独处理：跳过该组，避免点云/树信息干扰
        if parent is None:
            for sibling, sibling_lvl, sibling_parent in siblings_data:
                node_id = str(sibling.get("id"))
                gt_cands = _as_name_candidates(sibling)
                gt_name = gt_cands[0] if gt_cands else f"part_{node_id}"
                final_name = str(sibling.get("final_name") or sibling.get("_final_name") or gt_name)
                results.append(
                    NodeResult(
                        node_id=node_id,
                        level=sibling_lvl,
                        parent_id=None,
                        gt_name=gt_name,
                        final_name=final_name,
                        caption=str(sibling.get("caption") or ""),
                        caption_json=node_caption_cache.get(node_id, {"name": final_name, "caption": str(sibling.get("caption") or "")}),
                    )
                )
            continue

        # 获取父节点信息
        if parent is None:
            parent_name = ""
        else:
            parent_name = str(parent.get("_final_name") or (_as_name_candidates(parent)[0] if _as_name_candidates(parent) else ""))

        # 收集所有兄弟节点的索引和颜色
        sibling_indices_list: List[np.ndarray] = []
        sibling_colors: List[Tuple[int, int, int]] = []
        sibling_names: List[str] = []

        for i, sibling in enumerate(siblings):
            # 关键：不要直接使用 sibling["_computed_region_indices"]。
            # 因为 `_compute_region_points_bottom_up_partnext()` 可能为了控内存/控渲染对其做过随机下采样，
            # 从而导致 siblings 渲染时“应属于某兄弟节点的点”没有被任何 sibling_indices 覆盖，显示成灰点。
            #
            # 这里改为按叶子 maskId 精确回收该 sibling 覆盖的全量索引（基于 part_indices_cache 的连续索引段），
            # 可保证当兄弟节点集合逻辑上覆盖整个物体时，不会出现不该有的灰色“漏点”。
            indices_parts: List[np.ndarray] = []
            for mid in _collect_descendant_mask_ids_partnext(sibling):
                arr = part_indices_cache.get(str(mid))
                if isinstance(arr, np.ndarray) and len(arr) > 0:
                    indices_parts.append(arr.astype(np.int64, copy=False))
            indices = np.concatenate(indices_parts, axis=0) if indices_parts else np.array([], dtype=np.int64)

            if len(indices) > 0:
                sibling_indices_list.append(indices)
                sibling_colors.append(tuple(DISTINCT_COLORS[i % len(DISTINCT_COLORS)]))

            gt_cands = _as_name_candidates(sibling)
            gt_name = gt_cands[0] if gt_cands else f"part_{sibling.get('id', 'unknown')}"
            sibling_names.append(gt_name)

        # 渲染所有兄弟节点（SoM 风格：每个兄弟不同颜色）
        imgs = render_siblings_multiview(
            renderer=renderer,
            viewpoints=viewpoints,
            image_size=image_size,
            all_points=full_pointcloud if len(full_pointcloud) > 0 else None,
            sibling_indices_list=sibling_indices_list,
            sibling_colors=sibling_colors,
            model=model,
        )

        # 保存图像（文件名用兄弟组标识）
        sibling_ids = "_".join([str(s.get("id", "?")) for s in siblings[:3]])
        if len(siblings) > 3:
            sibling_ids += f"_etc{len(siblings)}"

        if save_images:
            for i, img in enumerate(imgs):
                Image.fromarray(img).save(os.path.join(images_dir, f"siblings_{sibling_ids}_L{lvl}_V{i}.png"))

        # Color name list (aligned with DISTINCT_COLORS)
        COLOR_NAMES = [
            "red",
            "green",
            "blue",
            "yellow",
            "magenta",
            "cyan",
            "orange",
            "violet",
            "spring green",
            "rose",
            "lime",
            "azure",
            "dark red",
            "dark green",
            "dark blue",
            "olive",
            "purple",
            "teal",
            "brown",
            "gold",
        ]
        sibling_color_names = [COLOR_NAMES[i % len(COLOR_NAMES)] for i in range(len(siblings))]

        # 一次性为所有兄弟节点生成 caption
        if dry_run or mllm_client is None:
            # Dry run 模式：生成占位 caption
            caption_results = []
            for idx, sibling in enumerate(siblings):
                gt_cands = _as_name_candidates(sibling)
                gt_name = gt_cands[0] if gt_cands else f"part_{sibling.get('id', 'unknown')}"
                caption_results.append({
                    "name": gt_name,
                    "caption": f"[dry_run] Placeholder caption for {gt_name} (color={sibling_color_names[idx]}).",
                    "color": sibling_color_names[idx],
                })
        else:
            # 调用 MLLM 一次性为所有兄弟生成 caption
            caption_results = name_and_caption_siblings(
                client=mllm_client,
                global_name=global_name,
                parent_name=parent_name,
                siblings=siblings,
                color_names=sibling_color_names,
                images=imgs,
                debug_save_dir=output_dir,
                debug_group_id=f"L{lvl}_{sibling_ids}",
            )

        # 将结果写入每个兄弟节点
        for idx, (sibling, sibling_lvl, sibling_parent) in enumerate(siblings_data):
            node_id = str(sibling.get("id"))
            gt_cands = _as_name_candidates(sibling)
            gt_name = gt_cands[0] if gt_cands else f"part_{node_id}"

            cap_obj = caption_results[idx] if idx < len(caption_results) else {"name": gt_name, "caption": ""}
            final_name = str(cap_obj.get("name") or gt_name)
            sibling["_final_name"] = final_name
            # 为 QA 生成提前“物化”字段：tree_summary / part_prompt 会读取 final_name/caption
            # 注意：不要在这里清理 _computed_region_indices，否则会影响后续 QA 的高亮渲染。
            sibling["final_name"] = final_name
            sibling["caption"] = str(cap_obj.get("caption") or "")

            if sibling_parent is None:
                global_name = final_name
                global_caption = str(cap_obj.get("caption") or "")

            node_caption_cache[node_id] = cap_obj
            results.append(
                NodeResult(
                    node_id=node_id,
                    level=sibling_lvl,
                    parent_id=parent_id,
                    gt_name=gt_name,
                    final_name=final_name,
                    caption=str(cap_obj.get("caption") or ""),
                    caption_json=cap_obj,
                )
            )

    renderer.cleanup()

    # ========== QA 生成（可选）：基于部件树 caption + 多视角图 ==========
    qa_info: Optional[Dict[str, Any]] = None
    if generate_qa and (not dry_run) and mllm_client is not None:
        try:
            qa_items, qa_debug = generate_partnext_qa_dataset(
                client=mllm_client,
                object_id=object_id,
                tree=tree,
                glb_path=glb_path,
                image_size=image_size,
                viewpoints=viewpoints,
                full_pointcloud=full_pointcloud,
                qa_part_nodes=qa_part_nodes,
                qa_max_parts=qa_max_parts,
                qa_num_other=qa_num_other,
                qa_enable_part_count=qa_enable_part_count,
                qa_num_part_count=qa_num_part_count,
                qa_max_points=qa_max_points,
                qa_max_tokens=qa_max_tokens,
                qa_temperature=qa_temperature,
                seed=seed,
                qa_debug_print_prompts=qa_debug_print_prompts,
                qa_debug_print_responses=qa_debug_print_responses,
            )
            if qa_output_path is None:
                qa_output_path = os.path.join(output_dir, f"{object_id}_qa.jsonl")
            os.makedirs(os.path.dirname(qa_output_path), exist_ok=True)
            with open(qa_output_path, "w", encoding="utf-8") as f:
                for it in qa_items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            qa_debug_path = os.path.join(output_dir, f"{object_id}_qa_debug.json")
            with open(qa_debug_path, "w", encoding="utf-8") as f:
                json.dump(qa_debug, f, ensure_ascii=False, indent=2)

            qa_info = {"qa_output_path": qa_output_path, "qa_debug_path": qa_debug_path, "num_qas": len(qa_items)}
            print(f"[QA] 完成：qas={len(qa_items)}，写入 {qa_output_path}")
        except Exception as e:
            print(f"[QA] 生成失败（跳过）：{e}")

    def attach(node: Dict[str, Any]):
        nid = str(node.get("id"))
        cap = node_caption_cache.get(nid)
        if cap:
            node["final_name"] = node.get("_final_name", cap.get("name", node.get("name")))
            node["caption"] = cap.get("caption", "")
            node["caption_struct"] = cap
        for c in _collect_children(node):
            attach(c)
        node.pop("_computed_region_points", None)
        node.pop("_computed_region_indices", None)
        node.pop("_final_name", None)

    attach(tree)

    out = {
        "object_id": object_id,
        "glb_path": glb_path,
        "statistics": {"num_nodes": len(results)},
        "tree": tree,
        "flat_nodes": [asdict(r) for r in results],
    }
    if qa_info:
        out["qa"] = qa_info

    out_path = os.path.join(output_dir, f"{object_id}_partnext_tree_caption.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return {"output_path": out_path, "num_nodes": len(results)}


def main():
    p = argparse.ArgumentParser(description="PartNeXt hierarchyList 逐层命名与 caption 生成")
    p.add_argument("--partnext_glb_dir", type=str, required=True, help="PartNeXt GLB 根目录（如 data/data）")
    p.add_argument("--partnext_ann_dir", type=str, required=True, help="PartNeXt 标注目录（load_from_disk 的目录）")
    p.add_argument("--partnext_object_id", type=str, required=True, help="PartNeXt model_id")
    p.add_argument("--partnext_part_sample_points", type=int, default=5000, help="每个叶子部件采样点数（用于高亮）")
    p.add_argument("--output_dir", type=str, default="outputs/partnext_tree_captions", help="输出目录")
    p.add_argument("--image_size", type=int, default=800, help="渲染分辨率（正方形）")
    p.add_argument("--num_views", type=int, default=6, help="每个节点使用的视角数（<=6）")
    p.add_argument("--view_radius", type=float, default=1.5, help="相机半径（建议 1.2~2.0）")
    p.add_argument("--traversal", type=str, default="bfs", choices=["bfs", "dfs"], help="遍历顺序")
    p.add_argument("--save_images", action="store_true", default=True, help="保存每个节点的输入图像")
    p.add_argument("--dry_run", action="store_true", default=False, help="不调用 MLLM，仅跑渲染与占位输出")

    # QA 生成（可选）
    p.add_argument("--generate_qa", action="store_true", default=False, help="基于部件树 caption + 多视角图生成 QA 数据（jsonl）")
    p.add_argument("--qa_output_path", type=str, default=None, help="QA 输出 jsonl 路径（默认写入 output_dir/{object_id}_qa.jsonl）")
    p.add_argument("--qa_part_nodes", type=str, default="leaf", choices=["leaf", "all"], help="用于生成部件 QA 的节点集合")
    p.add_argument("--qa_max_parts", type=int, default=25, help="最多生成多少条部件 QA（按遍历顺序截断）")
    p.add_argument("--qa_num_other", type=int, default=6, help="其他形式 QA 最大数量（模型会自适应生成 0..N 条，不再强行固定 N 条）")
    p.add_argument("--qa_num_part_count", type=int, default=3, help="部件计数 QA 最大数量（模型会自适应生成 0..N 条；不适合可输出空 qas）")
    p.add_argument(
        "--no_qa_part_count",
        action="store_false",
        dest="qa_enable_part_count",
        default=True,
        help="关闭“部件计数”QA（默认开启；若树太简单/不适合，模型也可能输出空 qas）",
    )
    p.add_argument("--qa_max_points", type=int, default=200000, help="QA 高亮渲染时点云最多使用多少点（过大会很慢）")
    p.add_argument("--qa_max_tokens", type=int, default=8192*2, help="QA 生成 max_tokens")
    p.add_argument("--qa_temperature", type=float, default=0.7, help="QA 生成 temperature")
    p.add_argument("--seed", type=int, default=0, help="随机种子（影响点云下采样等）")
    p.add_argument(
        "--no_qa_debug_print_prompts",
        action="store_false",
        dest="qa_debug_print_prompts",
        default=True,
        help="关闭 QA prompt 的 stdout 打印（prompt 仍会写入 qa_debug.json）",
    )
    p.add_argument(
        "--qa_debug_print_responses",
        action="store_true",
        default=False,
        help="将 QA 的模型原始回答打印到 stdout（用于 debug；raw 也会写入 qa_debug.json）",
    )

    # MLLM
    p.add_argument("--mllm_provider", type=str, default="dashscope", choices=["openai", "anthropic", "openai-compatible", "dashscope"])
    p.add_argument("--mllm_api_key", type=str, default=None, help="API Key（或使用环境变量）")
    p.add_argument("--mllm_model", type=str, default=None, help="模型名（dashscope 默认 qwen3-vl-plus）")
    p.add_argument("--mllm_base_url", type=str, default=None, help="自定义 base_url（openai-compatible/dashscope）")
    p.add_argument("--enable_thinking", action="store_true", default=False, help="dashscope 思考模式（一般不需要）")
    p.add_argument("--thinking_budget", type=int, default=4096, help="dashscope 思考 token 上限")

    args = p.parse_args()

    if args.dry_run:
        client = None
    else:
        api_key = args.mllm_api_key
        if api_key is None:
            if args.mllm_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif args.mllm_provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif args.mllm_provider == "dashscope":
                api_key = os.environ.get("DASHSCOPE_API_KEY")
            else:
                api_key = os.environ.get("MLLM_API_KEY")
        if not api_key:
            raise SystemExit("缺少 API Key：请用 --mllm_api_key 或设置对应环境变量（或使用 --dry_run）")
        client = create_mllm_client(
            args.mllm_provider,
            api_key,
            args.mllm_model,
            args.mllm_base_url,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )

    info = process_partnext_captioning(
        glb_dir=args.partnext_glb_dir,
        ann_dir=args.partnext_ann_dir,
        object_id=args.partnext_object_id,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_views=args.num_views,
        view_radius=args.view_radius,
        traversal=args.traversal,
        save_images=args.save_images,
        dry_run=args.dry_run,
        mllm_client=client,
        part_sample_points=args.partnext_part_sample_points,
        generate_qa=args.generate_qa,
        qa_output_path=args.qa_output_path,
        qa_part_nodes=args.qa_part_nodes,
        qa_max_parts=args.qa_max_parts,
        qa_num_other=args.qa_num_other,
        qa_enable_part_count=args.qa_enable_part_count,
        qa_num_part_count=args.qa_num_part_count,
        qa_max_points=args.qa_max_points,
        qa_max_tokens=args.qa_max_tokens,
        qa_temperature=args.qa_temperature,
        seed=args.seed,
        qa_debug_print_prompts=args.qa_debug_print_prompts,
        qa_debug_print_responses=args.qa_debug_print_responses,
    )
    print(f"完成：nodes={info['num_nodes']}，结果写入 {info['output_path']}")


if __name__ == "__main__":
    main()


