"""
PartNeXt QA 数据转换脚本

将 batch_partnext_qa.py 生成的 QA jsonl 转换为 PointLLM 训练格式。

输入:
    - QA jsonl 文件（merged_qa.jsonl 或单个 object 的 xxx_qa.jsonl）
    - PartNeXt 点云数据（从 GLB 或预处理的 .npy）

输出:
    - PointLLM 格式的标注 JSON 文件
    - 点云 .npy 文件（带可选的高亮蒙版）

数据格式说明:
    - object_identity / other / part_count: 点云 + prompt（无高亮）
    - part_identity: 点云 + prompt，其中对应部件的点被高亮（颜色通道修改）

使用方法:
    python convert_partnext_qa_to_pointllm.py \
        --qa_jsonl outputs/partnext_qa_batch/merged_qa.jsonl \
        --partnext_glb_dir /path/to/glbs \
        --partnext_ann_dir /path/to/annotations \
        --output_dir outputs/pointllm_format \
        --pointnum 8192

    # 或使用预处理的点云
    python convert_partnext_qa_to_pointllm.py \
        --qa_jsonl outputs/partnext_qa_batch/merged_qa.jsonl \
        --pointcloud_dir outputs/partnext_pointclouds \
        --output_dir outputs/pointllm_format \
        --pointnum 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# 添加项目路径
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 尝试导入 PartNeXt
try:
    PARTNEXT_ROOT = Path(__file__).resolve().parents[1] / "PartNeXt" / "PartNeXt_lib"
    if PARTNEXT_ROOT.exists() and str(PARTNEXT_ROOT) not in sys.path:
        sys.path.insert(0, str(PARTNEXT_ROOT))
    from partnext import PartNeXtDataset
    PARTNEXT_AVAILABLE = True
except Exception:
    PARTNEXT_AVAILABLE = False


# 高亮颜色常量（归一化到 0-1）
HIGHLIGHT_COLORS = [
    (1.0, 0.0, 0.0),   # 红色
    (0.0, 1.0, 0.0),   # 绿色
    (0.0, 0.0, 1.0),   # 蓝色
    (1.0, 1.0, 0.0),   # 黄色
    (1.0, 0.0, 1.0),   # 洋红
    (0.0, 1.0, 1.0),   # 青色
    (1.0, 0.5, 0.0),   # 橙色
    (0.5, 0.0, 1.0),   # 紫色
]

HIGHLIGHT_COLOR_NAMES = [
    "red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"
]


def pc_norm(pc: np.ndarray) -> np.ndarray:
    """点云归一化到单位球"""
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]
    
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if m > 0:
        xyz = xyz / m
    
    return np.concatenate((xyz, other_feature), axis=1)


def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """最远点采样"""
    N, D = point.shape
    if N <= npoint:
        # 如果点数不够，重复采样
        if N == 0:
            return np.zeros((npoint, D), dtype=point.dtype)
        indices = np.concatenate([
            np.arange(N),
            np.random.choice(N, npoint - N, replace=True)
        ])
        return point[indices]
    
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return point[centroids]


def apply_highlight_mask(
    point_cloud: np.ndarray,
    highlight_indices: np.ndarray,
    highlight_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    dim_factor: float = 0.4,
) -> np.ndarray:
    """
    在点云上应用高亮蒙版。
    
    对于 part_identity QA，高亮目标部件的点，并将其他点调暗。
    
    Args:
        point_cloud: (N, 6) 点云，前3列是 xyz，后3列是 rgb (0-1)
        highlight_indices: 需要高亮的点索引
        highlight_color: 高亮颜色 (r, g, b)，范围 0-1
        dim_factor: 非高亮点的亮度系数
    
    Returns:
        (N, 6) 修改后的点云
    """
    pc = point_cloud.copy()
    N = pc.shape[0]
    
    # 确保点云有颜色通道
    if pc.shape[1] < 6:
        # 如果没有颜色，添加默认灰色
        colors = np.ones((N, 3)) * 0.7
        pc = np.concatenate([pc[:, :3], colors], axis=1)
    
    # 创建高亮蒙版
    highlight_mask = np.zeros(N, dtype=bool)
    valid_indices = highlight_indices[highlight_indices < N]
    highlight_mask[valid_indices] = True
    
    # 调暗非高亮点
    pc[~highlight_mask, 3:6] *= dim_factor
    
    # 高亮目标点（混合原色和高亮色）
    blend_factor = 0.7  # 高亮色权重
    for i, c in enumerate(highlight_color):
        pc[highlight_mask, 3 + i] = (
            pc[highlight_mask, 3 + i] * (1 - blend_factor) + c * blend_factor
        )
    
    return pc


def sample_pointcloud_from_mesh(
    mesh,
    num_samples: int = 8192,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 trimesh Scene/Mesh 采样点云。
    
    Returns:
        points: (N, 3) 点坐标
        colors: (N, 3) 颜色 (0-1)
        face_indices: (N,) 每个点对应的面索引
    """
    import trimesh
    
    np.random.seed(seed)
    
    # 如果是 Scene，合并所有 mesh
    if isinstance(mesh, trimesh.Scene):
        mesh_list = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not mesh_list:
            return None, None, None
        merged_mesh = trimesh.util.concatenate(mesh_list)
    elif isinstance(mesh, trimesh.Trimesh):
        merged_mesh = mesh
    else:
        return None, None, None
    
    # 采样点
    points, face_indices = merged_mesh.sample(num_samples, return_index=True)
    
    # 获取颜色
    if hasattr(merged_mesh.visual, 'face_colors') and merged_mesh.visual.face_colors is not None:
        colors = merged_mesh.visual.face_colors[face_indices, :3] / 255.0
    elif hasattr(merged_mesh.visual, 'vertex_colors') and merged_mesh.visual.vertex_colors is not None:
        # 使用面的顶点颜色平均
        face_vertex_colors = merged_mesh.visual.vertex_colors[merged_mesh.faces[face_indices]]
        colors = face_vertex_colors.mean(axis=1)[:, :3] / 255.0
    else:
        # 默认灰色
        colors = np.ones((num_samples, 3)) * 0.7
    
    return points.astype(np.float32), colors.astype(np.float32), face_indices


def get_all_node_mask_ids(hierarchy_list: List[Dict]) -> Dict[str, List[int]]:
    """
    从层级列表中收集每个节点的所有 mask ID（包括子节点）。
    
    Returns:
        {node_id: [mask_id, ...]} 字典
    """
    all_node_masks = {}
    
    def get_mask_ids(node: Dict) -> List[int]:
        """递归收集节点及其所有子节点的 mask ID"""
        mask_ids = []
        if "maskId" in node:
            mask_ids.append(node["maskId"])
        if "children" in node:
            for child in node["children"]:
                mask_ids.extend(get_mask_ids(child))
        return mask_ids
    
    def collect_nodes(node: Dict):
        """递归收集所有节点"""
        node_id = str(node.get("nodeId", ""))
        ref_node_id = node.get("refNodeId", -1)
        
        # 跳过根节点 (refNodeId == 0 或 -1)
        if ref_node_id not in (0, -1) and node_id:
            all_node_masks[node_id] = get_mask_ids(node)
        
        if "children" in node:
            for child in node["children"]:
                collect_nodes(child)
    
    # 处理所有顶层节点
    for root in hierarchy_list:
        collect_nodes(root)
    
    return all_node_masks


def load_partnext_pointcloud(
    dataset: "PartNeXtDataset",
    object_id: str,
    pointnum: int = 8192,
    use_color: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    从 PartNeXt 加载物体点云和部件索引。
    
    Returns:
        point_cloud: (N, 6) 或 (N, 3) 点云
        part_indices: {node_id: indices} 字典
    """
    try:
        # 加载 PartNeXtObject
        pn_object = dataset.load_object(object_id)
        if pn_object is None:
            print(f"物体不存在: {object_id}")
            return None, {}
        
        # 从 mesh 采样点云
        mesh = pn_object.get_mesh()
        points, colors, face_indices = sample_pointcloud_from_mesh(mesh, pointnum, seed)
        
        if points is None:
            print(f"采样点云失败: {object_id}")
            return None, {}
        
        # 合并 xyz 和颜色
        full_pc = np.concatenate([points, colors], axis=1)
        
        # 获取层级结构和 masks
        hierarchy_list = pn_object.hierarchyList  # List[Dict]
        masks = pn_object.masks  # {mask_id: {mesh_idx: face_indices}}
        
        # 计算每个 mesh 的 face 偏移量
        mesh_face_offsets = {}
        offset = 0
        geometry_list = pn_object.geometry_list
        for i, geom in enumerate(geometry_list):
            mesh_face_offsets[i] = offset
            if hasattr(geom, 'faces'):
                offset += len(geom.faces)
        
        # 收集每个节点的 mask ID
        all_node_masks = get_all_node_mask_ids(hierarchy_list)
        
        # 计算每个节点对应的点索引
        part_indices = {}
        
        for node_id, mask_ids in all_node_masks.items():
            # 收集该节点所有 mask 对应的 face 索引
            node_face_indices = []
            for mask_id in mask_ids:
                mask_key = str(mask_id)
                if mask_key in masks:
                    for mesh_idx_str, face_list in masks[mask_key].items():
                        mesh_idx = int(mesh_idx_str)
                        face_offset = mesh_face_offsets.get(mesh_idx, 0)
                        node_face_indices.extend([f + face_offset for f in face_list])
            
            if node_face_indices:
                # 找到采样点中属于这些 face 的点
                node_face_set = set(node_face_indices)
                point_mask = np.array([fi in node_face_set for fi in face_indices])
                indices = np.where(point_mask)[0]
                
                if len(indices) > 0:
                    part_indices[node_id] = indices
        
        # 归一化
        full_pc = pc_norm(full_pc)
        
        if not use_color:
            full_pc = full_pc[:, :3]
        
        return full_pc, part_indices
        
    except Exception as e:
        import traceback
        print(f"加载 {object_id} 失败: {e}")
        traceback.print_exc()
        return None, {}


def load_precomputed_pointcloud(
    pointcloud_dir: str,
    object_id: str,
    pointnum: int = 8192,
) -> Optional[np.ndarray]:
    """从预计算的 .npy 文件加载点云"""
    filename = f"{object_id}_{pointnum}.npy"
    filepath = os.path.join(pointcloud_dir, filename)
    
    if os.path.exists(filepath):
        pc = np.load(filepath)
        return pc_norm(pc)
    
    # 尝试不带点数后缀的文件名
    filename = f"{object_id}.npy"
    filepath = os.path.join(pointcloud_dir, filename)
    if os.path.exists(filepath):
        pc = np.load(filepath)
        if len(pc) != pointnum:
            pc = farthest_point_sample(pc, pointnum)
        return pc_norm(pc)
    
    return None


def generate_pointllm_prompt(qa_item: Dict[str, Any], highlight_info: Optional[str] = None) -> str:
    """
    生成 PointLLM 格式的 prompt。
    
    Args:
        qa_item: QA 数据项
        highlight_info: 高亮信息描述（用于 part_identity）
    
    Returns:
        格式化的 prompt 字符串
    """
    qa_type = qa_item.get("qa_type", "")
    question = qa_item.get("question", "What is this 3D object?")
    
    # 基础 prompt
    prompt = f"<point>\n{question}"
    
    # 对于 part_identity，添加高亮说明
    if qa_type == "part_identity" and highlight_info:
        prompt = f"<point>\nNote: The target part is highlighted in {highlight_info} color in the point cloud.\n{question}"
    
    return prompt


def convert_qa_to_pointllm_format(
    qa_items: List[Dict[str, Any]],
    object_id: str,
    base_pointcloud: np.ndarray,
    part_indices: Dict[str, np.ndarray],
    output_pc_dir: str,
    pointnum: int = 8192,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    将 QA 数据转换为 PointLLM 格式。
    
    对于 part_identity QA，生成带高亮蒙版的点云文件。
    对于其他类型 QA，使用原始点云。
    
    Returns:
        PointLLM 格式的标注列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    annotations = []
    
    for idx, qa in enumerate(qa_items):
        qa_type = qa.get("qa_type", "other")
        node_id = qa.get("node_id")
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        think = qa.get("think", "")
        
        # 确定点云文件名和是否需要高亮
        if qa_type == "part_identity" and node_id and node_id in part_indices:
            # 为 part_identity 创建带高亮的点云
            highlight_idx = hash(f"{object_id}_{node_id}") % len(HIGHLIGHT_COLORS)
            highlight_color = HIGHLIGHT_COLORS[highlight_idx]
            highlight_color_name = HIGHLIGHT_COLOR_NAMES[highlight_idx]
            
            part_idx = part_indices[node_id]
            highlighted_pc = apply_highlight_mask(
                base_pointcloud,
                part_idx,
                highlight_color=highlight_color,
            )
            
            # 保存高亮点云
            pc_filename = f"{object_id}_part_{node_id}_{pointnum}.npy"
            pc_path = os.path.join(output_pc_dir, pc_filename)
            np.save(pc_path, highlighted_pc.astype(np.float32))
            
            # 生成 prompt（带高亮说明）
            prompt = generate_pointllm_prompt(qa, highlight_info=highlight_color_name)
            
            # 记录使用的 object_id（用于加载点云）
            # 格式：{object_id}_part_{node_id}
            pc_object_id = f"{object_id}_part_{node_id}"
        else:
            # 其他类型使用原始点云
            pc_filename = f"{object_id}_{pointnum}.npy"
            pc_path = os.path.join(output_pc_dir, pc_filename)
            
            # 只在第一次保存原始点云
            if not os.path.exists(pc_path):
                np.save(pc_path, base_pointcloud.astype(np.float32))
            
            prompt = generate_pointllm_prompt(qa)
            pc_object_id = object_id
        
        # 构建 PointLLM 格式的标注
        # 根据是否有 think 字段，生成不同格式的回答
        if think:
            response = f"<think>{think}</think><answer>{answer}</answer>"
        else:
            response = answer
        
        annotation = {
            "object_id": pc_object_id,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": response}
            ],
            "conversation_type": f"partnext_{qa_type}",
            # 保留原始元数据
            "meta": {
                "original_object_id": object_id,
                "qa_type": qa_type,
                "node_id": node_id,
                "node_name": qa.get("node_name"),
            }
        }
        
        annotations.append(annotation)
    
    return annotations


def main():
    parser = argparse.ArgumentParser(description="转换 PartNeXt QA 到 PointLLM 格式")
    
    # 输入
    parser.add_argument("--qa_jsonl", type=str, required=True,
                        help="QA jsonl 文件路径")
    parser.add_argument("--partnext_glb_dir", type=str, default=None,
                        help="PartNeXt GLB 目录（与 pointcloud_dir 二选一）")
    parser.add_argument("--partnext_ann_dir", type=str, default=None,
                        help="PartNeXt 标注目录")
    parser.add_argument("--pointcloud_dir", type=str, default=None,
                        help="预计算点云目录（与 partnext_glb_dir 二选一）")
    
    # 输出
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    
    # 参数
    parser.add_argument("--pointnum", type=int, default=8192,
                        help="点云点数")
    parser.add_argument("--use_color", action="store_true", default=True,
                        help="使用颜色信息")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="训练/验证分割比例")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.qa_jsonl):
        print(f"错误: QA 文件不存在: {args.qa_jsonl}")
        sys.exit(1)
    
    use_partnext = args.partnext_glb_dir is not None and args.partnext_ann_dir is not None
    use_precomputed = args.pointcloud_dir is not None
    
    if not use_partnext and not use_precomputed:
        print("错误: 必须指定 --partnext_glb_dir + --partnext_ann_dir 或 --pointcloud_dir")
        sys.exit(1)
    
    # 创建输出目录
    output_pc_dir = os.path.join(args.output_dir, "pointclouds")
    output_anno_dir = os.path.join(args.output_dir, "anno_data")
    os.makedirs(output_pc_dir, exist_ok=True)
    os.makedirs(output_anno_dir, exist_ok=True)
    
    # 加载 QA 数据
    print(f"加载 QA 数据: {args.qa_jsonl}")
    qa_by_object: Dict[str, List[Dict]] = defaultdict(list)
    with open(args.qa_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qa = json.loads(line)
                object_id = qa.get("object_id", "")
                if object_id:
                    qa_by_object[object_id].append(qa)
            except json.JSONDecodeError:
                continue
    
    print(f"共 {len(qa_by_object)} 个物体，{sum(len(v) for v in qa_by_object.values())} 条 QA")
    
    # 初始化 PartNeXt（如果使用）
    dataset = None
    if use_partnext:
        if not PARTNEXT_AVAILABLE:
            print("错误: PartNeXt 不可用")
            sys.exit(1)
        dataset = PartNeXtDataset(args.partnext_glb_dir, args.partnext_ann_dir)
    
    # 转换数据
    all_annotations = []
    failed_objects = []
    
    for object_id, qa_items in tqdm(qa_by_object.items(), desc="转换数据"):
        # 加载点云
        if use_partnext:
            base_pc, part_indices = load_partnext_pointcloud(
                dataset, object_id, args.pointnum, args.use_color
            )
        else:
            base_pc = load_precomputed_pointcloud(
                args.pointcloud_dir, object_id, args.pointnum
            )
            # 预计算模式下没有部件索引
            part_indices = {}
        
        if base_pc is None:
            failed_objects.append(object_id)
            continue
        
        # 确保点云有颜色通道
        if base_pc.shape[1] < 6 and args.use_color:
            colors = np.ones((len(base_pc), 3)) * 0.7
            base_pc = np.concatenate([base_pc[:, :3], colors], axis=1)
        
        # 转换
        annotations = convert_qa_to_pointllm_format(
            qa_items,
            object_id,
            base_pc,
            part_indices,
            output_pc_dir,
            args.pointnum,
            args.seed,
        )
        all_annotations.extend(annotations)
    
    print(f"\n转换完成: {len(all_annotations)} 条标注")
    if failed_objects:
        print(f"失败物体: {len(failed_objects)} 个")
    
    # 分割训练/验证集
    random.seed(args.seed)
    random.shuffle(all_annotations)
    
    split_idx = int(len(all_annotations) * args.split_ratio)
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]
    
    # 保存标注文件
    train_path = os.path.join(output_anno_dir, "partnext_qa_train.json")
    val_path = os.path.join(output_anno_dir, "partnext_qa_val.json")
    all_path = os.path.join(output_anno_dir, "partnext_qa_all.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_annotations, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_annotations, f, ensure_ascii=False, indent=2)
    
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    stats = defaultdict(int)
    for ann in all_annotations:
        qa_type = ann.get("meta", {}).get("qa_type", "unknown")
        stats[qa_type] += 1
    
    print(f"\n统计:")
    print(f"  训练集: {len(train_annotations)}")
    print(f"  验证集: {len(val_annotations)}")
    print(f"  QA 类型分布:")
    for qa_type, count in sorted(stats.items()):
        print(f"    {qa_type}: {count}")
    
    print(f"\n输出文件:")
    print(f"  点云目录: {output_pc_dir}")
    print(f"  训练标注: {train_path}")
    print(f"  验证标注: {val_path}")
    print(f"  全部标注: {all_path}")
    
    # 创建软链接说明
    print(f"\n使用说明:")
    print(f"  1. 在 PointLLM 数据目录创建软链接:")
    print(f"     ln -s {os.path.abspath(output_pc_dir)} /path/to/Point-R1/data/partnext_data")
    print(f"  2. 将标注文件复制到 anno_data 目录")
    print(f"  3. 修改训练脚本中的 data_path 和 anno_path")


if __name__ == "__main__":
    main()

