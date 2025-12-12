"""
交互式重新标注脚本

功能：
1. 读取已有的标注结果文件
2. 指定某个层级的父节点，复现其子簇的标注过程
3. 允许自定义 prompt 模板
4. 重新调用 MLLM API 获取新的标注

使用方法：
    # 基本用法：指定父节点层级和ID进行重新标注
    python replay_captioning.py \
        --caption_json outputs/captions/xxx_caption.json \
        --parent_level 0 \
        --parent_cluster_id 0 \
        --glb_path example_material/glbs/xxx.glb \
        --npy_path example_material/npys/xxx_8192.npy
    
    # 只生成 prompt，不调用 API（用于调试）
    python replay_captioning.py \
        --caption_json outputs/captions/xxx_caption.json \
        --parent_level 0 \
        --parent_cluster_id 0 \
        --glb_path example_material/glbs/xxx.glb \
        --npy_path example_material/npys/xxx_8192.npy \
        --dry_run
    
    # 使用自定义 prompt 文件
    python replay_captioning.py \
        --caption_json outputs/captions/xxx_caption.json \
        --parent_level 0 \
        --parent_cluster_id 0 \
        --glb_path example_material/glbs/xxx.glb \
        --npy_path example_material/npys/xxx_8192.npy \
        --visual_prompt_file custom_visual_prompt.txt \
        --som_prompt_file custom_som_prompt.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict
import time

import numpy as np
from PIL import Image

# 复用 hierarchical_captioning 中的模块
from hierarchical_captioning import (
    # 数据结构
    ClusterCaption,
    ViewQualityScore,
    ClusterAnnotation,
    ViewAnnotationResult,
    # MLLM 客户端
    MLLMClient,
    create_mllm_client,
    # 核心函数
    load_and_prepare_data,
    generate_sibling_views,
    get_view_direction_description,
    merge_multi_view_annotations,
    # Prompt 模板
    VISUAL_ANALYSIS_PROMPT,
    SOM_MATCHING_PROMPT,
    MERGE_ANNOTATIONS_PROMPT,
)


def convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型（包括字典键）"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # 同时转换键和值
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


# ===================== 辅助函数 =====================

def load_caption_json(json_path: str) -> Dict:
    """加载已有的标注结果"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_node_in_hierarchy(hierarchy: Dict, level: int, cluster_id: int) -> Optional[Dict]:
    """在层级结构中查找指定节点"""
    if hierarchy['level'] == level and hierarchy['cluster_id'] == cluster_id:
        return hierarchy
    
    for child in hierarchy.get('children', []):
        result = find_node_in_hierarchy(child, level, cluster_id)
        if result is not None:
            return result
    
    return None


def get_parent_context(hierarchy: Dict, target_level: int, target_cluster_id: int) -> Tuple[str, str, str, str]:
    """
    获取父节点上下文信息
    
    返回: (global_name, global_caption, parent_name, parent_caption)
    """
    global_name = hierarchy['name']
    global_caption = hierarchy['caption']
    
    # 查找目标节点
    node = find_node_in_hierarchy(hierarchy, target_level, target_cluster_id)
    
    if node is None:
        return global_name, global_caption, global_name, global_caption
    
    return global_name, global_caption, node['name'], node['caption']


def print_hierarchy_tree(hierarchy: Dict, indent: int = 0):
    """打印层级树结构"""
    prefix = "  " * indent
    level = hierarchy['level']
    cluster_id = hierarchy['cluster_id']
    name = hierarchy['name']
    point_count = hierarchy.get('point_count', '?')
    num_children = len(hierarchy.get('children', []))
    
    print(f"{prefix}[L{level}_C{cluster_id}] {name} ({point_count} pts, {num_children} children)")
    
    for child in hierarchy.get('children', []):
        print_hierarchy_tree(child, indent + 1)


def load_custom_prompt(file_path: Optional[str], default_prompt: str) -> str:
    """加载自定义 prompt 文件，如果不存在则使用默认值"""
    if file_path is None:
        return default_prompt
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"警告: 自定义 prompt 文件不存在: {file_path}，使用默认 prompt")
        return default_prompt


def save_prompt_templates(output_dir: str):
    """保存默认 prompt 模板到文件，方便用户修改"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'visual_analysis_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(VISUAL_ANALYSIS_PROMPT)
    
    with open(os.path.join(output_dir, 'som_matching_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(SOM_MATCHING_PROMPT)
    
    with open(os.path.join(output_dir, 'merge_annotations_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(MERGE_ANNOTATIONS_PROMPT)
    
    print(f"已保存 prompt 模板到: {output_dir}")
    print("  - visual_analysis_prompt.txt")
    print("  - som_matching_prompt.txt")
    print("  - merge_annotations_prompt.txt")


# ===================== 自定义标注函数 =====================

def call_mllm_for_sibling_annotation_custom(
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
    color_map: Dict[int, List[int]],
    visual_prompt_template: str,
    som_prompt_template: str,
    dry_run: bool = False
) -> Tuple[ViewAnnotationResult, Dict]:
    """
    调用 MLLM 获取单个视角的标注（支持自定义 prompt）
    """
    view_idx = view_info.get('view_idx', '?')
    
    # 1. 准备基础信息
    azimuth = view_info['azimuth']
    elevation = view_info['elevation']
    view_direction = get_view_direction_description(azimuth, elevation)
    view_info_str = f"- 视角: {view_direction}"
    
    # ================= Step 1: 视觉分析 (基于 Clean View) =================
    
    analysis_prompt = visual_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        current_level=current_level,
        view_info=view_info_str
    )
    
    print(f"\n{'='*20} PROMPT (Step 1: Visual Analysis - View {view_idx}) {'='*20}", flush=True)
    print(analysis_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    if dry_run:
        print(f"[DRY RUN] 跳过 Step 1 MLLM 调用")
        analysis_response = "[DRY RUN] 模拟视觉分析结果"
    else:
        content_step1 = [
            {"type": "text", "text": analysis_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(clean_image)}"}}
        ]
        messages_step1 = [{"role": "user", "content": content_step1}]
        analysis_response = client.call(messages_step1, enable_thinking=False)
    
    print(f"\n{'='*20} RESPONSE (Step 1) {'='*20}", flush=True)
    print(analysis_response, flush=True)
    print(f"{'='*60}\n", flush=True)

    # ================= Step 2: 匹配与标注 (基于 SoM View) =================

    # 生成颜色映射说明
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
    
    color_mapping_lines = []
    for cid in child_ids:
        som_id = som_id_map[cid]
        rgb = tuple(color_map[cid])
        color_name = COLOR_NAMES.get(rgb, f"RGB{rgb}")
        color_mapping_lines.append(f"  - **编号 {som_id}**: {color_name} 区域")
    color_mapping_str = "\n".join(color_mapping_lines)
    
    matching_prompt = som_prompt_template.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        visual_analysis=analysis_response,
        color_mapping=color_mapping_str
    )
    
    print(f"\n{'='*20} PROMPT (Step 2: SoM Matching - View {view_idx}) {'='*20}", flush=True)
    print(matching_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry = {
        "type": "sibling_view_annotation",
        "view_idx": view_idx,
        "step1_prompt": analysis_prompt,
        "step1_response": analysis_response,
        "step2_prompt": matching_prompt,
        "input_images": ["clean_image (Step 1)", "som_image (Step 2)"],
        "output_response": ""
    }
    
    if dry_run:
        print(f"[DRY RUN] 跳过 Step 2 MLLM 调用")
        response = '{"quality_assessment": {"clarity_score": 8, "completeness_score": 7, "occlusion_score": 6, "distinguishability_score": 8, "overall_score": 7.25, "reasoning": "DRY RUN"}, "annotations": [], "unmatched_features": []}'
    else:
        content_step2 = [
            {"type": "text", "text": matching_prompt},
            {"type": "text", "text": "[图像: SoM 视图 (仅几何形状与区域编码，颜色为区域ID)]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(som_image)}"}}
        ]
        messages_step2 = [{"role": "user", "content": content_step2}]
        response = client.call(messages_step2, enable_thinking=True)
    
    print(f"\n{'='*20} RESPONSE (Step 2) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    log_entry["output_response"] = response
    
    # 解析结果
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            
            quality = result.get('quality_assessment', {})
            quality_score = ViewQualityScore(
                view_idx=view_info['view_idx'],
                clarity_score=quality.get('clarity_score', 5),
                completeness_score=quality.get('completeness_score', 5),
                occlusion_score=quality.get('occlusion_score', 5),
                distinguishability_score=quality.get('distinguishability_score', 5),
                overall_score=quality.get('overall_score', 5),
                reasoning=quality.get('reasoning', '')
            )
            
            annotations = []
            for ann in result.get('annotations', []):
                annotations.append(ClusterAnnotation(
                    cluster_id=child_ids[ann['som_id'] - 1] if ann['som_id'] <= len(child_ids) else -1,
                    som_id=ann['som_id'],
                    name=ann.get('name', ''),
                    description=ann.get('description', ''),
                    confidence=ann.get('confidence', 0.5),
                    color=ann.get('color', '未指定'),
                    matched_features=ann.get('matched_features', [])
                ))
            
            unmatched = result.get('unmatched_features', [])
            log_entry['parsed_result'] = {
                'quality_score': asdict(quality_score),
                'annotations': [asdict(a) for a in annotations],
                'unmatched_features': unmatched
            }
            
            return ViewAnnotationResult(
                view_idx=view_info['view_idx'],
                quality_score=quality_score,
                annotations=annotations,
                unmatched_features=unmatched
            ), log_entry
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"解析响应失败: {e}")
    
    # 默认返回
    return ViewAnnotationResult(
        view_idx=view_info['view_idx'],
        quality_score=ViewQualityScore(
            view_idx=view_info['view_idx'],
            clarity_score=5, completeness_score=5, occlusion_score=5,
            distinguishability_score=5, overall_score=5, reasoning="解析失败"
        ),
        annotations=[],
        unmatched_features=[]
    ), log_entry


# ===================== 主流程 =====================

def replay_captioning(
    caption_json_path: str,
    glb_path: str,
    npy_path: str,
    feature_path: str,
    parent_level: int,
    parent_cluster_id: int,
    output_dir: str,
    mllm_client: Optional[MLLMClient],
    visual_prompt_template: str,
    som_prompt_template: str,
    merge_prompt_template: str,
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.3, 0.5, 0.7],
    min_cluster_points: int = 100,
    save_images: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    重新执行指定节点的标注
    """
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    # 1. 加载已有的标注结果
    print(f"\n{'='*60}")
    print(f"加载已有标注结果: {caption_json_path}")
    print(f"{'='*60}\n")
    
    caption_data = load_caption_json(caption_json_path)
    hierarchy = caption_data['hierarchy']
    
    print("层级结构:")
    print_hierarchy_tree(hierarchy)
    print()
    
    # 2. 获取上下文信息
    global_name, global_caption, parent_name, parent_caption = get_parent_context(
        hierarchy, parent_level, parent_cluster_id
    )
    
    print(f"目标父节点: Level {parent_level}, Cluster {parent_cluster_id}")
    print(f"  全局名称: {global_name}")
    print(f"  父节点名称: {parent_name}")
    print(f"  父节点描述: {parent_caption[:100]}...")
    print()
    
    # 3. 加载数据和执行聚类
    print("[Step 1] 加载数据和执行聚类...")
    points, clustering_results, model = load_and_prepare_data(
        glb_path, npy_path, feature_path, k_neighbors, betas
    )
    
    # 4. 生成视图
    print(f"\n[Step 2] 生成 Level {parent_level}, Cluster {parent_cluster_id} 的子簇视图...")
    views, som_id_map, child_level_idx, msg = generate_sibling_views(
        points, clustering_results, model,
        parent_level, parent_cluster_id,
        image_size=800, num_views=4
    )
    
    if not views:
        print(f"错误: {msg}")
        return {"error": msg}
    
    child_ids = views[0]['child_ids']
    child_labels = clustering_results.get(child_level_idx)
    
    print(f"发现分裂: Level {parent_level} -> Level {child_level_idx} ({len(child_ids)} 个子簇)")
    
    # 过滤太小的簇
    valid_child_ids = []
    for cid in child_ids:
        count = np.sum(child_labels == cid)
        status = "✓ 保留" if count >= min_cluster_points else "✗ 被过滤"
        print(f"  - 簇 ID {cid}: {count} 点 -> {status}")
        if count >= min_cluster_points:
            valid_child_ids.append(cid)
    
    if len(valid_child_ids) <= 1:
        print(f"错误: 有效子簇数量不足 ({len(valid_child_ids)})")
        return {"error": "有效子簇数量不足"}
    
    # 5. 保存视图图像
    if save_images:
        for i, view in enumerate(views):
            if view['som_image'] is not None:
                # 保存 Clean 视图
                Image.fromarray(view['clean_image']).save(
                    os.path.join(images_dir, f"replay_L{parent_level}_C{parent_cluster_id}_clean_{i}.png")
                )
                # 保存 SoM 视图
                Image.fromarray(view['som_image']).save(
                    os.path.join(images_dir, f"replay_L{parent_level}_C{parent_cluster_id}_som_{i}.png")
                )
                # 保存拼接视图
                combined = np.concatenate([view['clean_image'], view['som_image']], axis=1)
                Image.fromarray(combined).save(
                    os.path.join(images_dir, f"replay_L{parent_level}_C{parent_cluster_id}_combined_{i}.png")
                )
        print(f"\n已保存视图图像到: {images_dir}")
    
    # 6. 多视角标注
    print(f"\n[Step 3] 执行多视角标注...")
    
    interaction_logs = []
    view_results = []
    
    for view in views:
        if view['som_image'] is None:
            continue
        
        result, log = call_mllm_for_sibling_annotation_custom(
            mllm_client,
            view['clean_image'],
            view['som_image'],
            global_name,
            global_caption,
            parent_name,
            parent_caption,
            child_level_idx,
            view['view_info'],
            valid_child_ids,
            som_id_map,
            view['color_map'],
            visual_prompt_template,
            som_prompt_template,
            dry_run=dry_run
        )
        view_results.append(result)
        interaction_logs.append(log)
        
        print(f"视角 {view['view_info']['view_idx']}: 质量={result.quality_score.overall_score:.1f}")
    
    # 7. 合并标注
    print(f"\n[Step 4] 合并多视角标注...")
    
    if dry_run:
        print("[DRY RUN] 跳过合并标注")
        merged_annotations = []
    else:
        merged_annotations, merge_log = merge_multi_view_annotations(
            mllm_client, view_results, views, global_name, parent_name,
            valid_child_ids, som_id_map
        )
        interaction_logs.append(merge_log)
    
    # 8. 保存结果
    processing_time = time.time() - start_time
    
    result = {
        "parent_level": parent_level,
        "parent_cluster_id": parent_cluster_id,
        "parent_name": parent_name,
        "parent_caption": parent_caption,
        "child_level": child_level_idx,
        "child_ids": valid_child_ids,
        "som_id_map": som_id_map,
        "merged_annotations": merged_annotations,
        "view_results": [
            {
                "view_idx": vr.view_idx,
                "quality_score": asdict(vr.quality_score),
                "annotations": [asdict(a) for a in vr.annotations],
                "unmatched_features": vr.unmatched_features
            }
            for vr in view_results
        ],
        "processing_time": processing_time
    }
    
    # 保存结果 JSON
    result_path = os.path.join(output_dir, f"replay_L{parent_level}_C{parent_cluster_id}_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(result), f, ensure_ascii=False, indent=2)
    
    # 保存交互日志
    log_path = os.path.join(output_dir, f"replay_L{parent_level}_C{parent_cluster_id}_interaction.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(interaction_logs), f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"重新标注完成!")
    print(f"  处理时间: {processing_time:.1f}s")
    print(f"  结果保存至: {result_path}")
    print(f"  交互日志: {log_path}")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='交互式重新标注脚本')
    
    # 输入文件
    parser.add_argument('--caption_json', type=str, required=True,
                       help='已有的标注结果 JSON 文件路径')
    parser.add_argument('--glb_path', type=str, required=True,
                       help='GLB 文件路径')
    parser.add_argument('--npy_path', type=str, required=True,
                       help='点云 NPY 文件路径')
    parser.add_argument('--feature_path', type=str, default=None,
                       help='特征 NPY 文件路径（如果未指定，将自动推断）')
    
    # 目标节点
    parser.add_argument('--parent_level', type=int, required=True,
                       help='父节点的层级索引')
    parser.add_argument('--parent_cluster_id', type=int, required=True,
                       help='父节点的簇 ID')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='outputs/replay',
                       help='输出目录')
    parser.add_argument('--save_images', action='store_true', default=True,
                       help='是否保存中间图像')
    
    # Prompt 自定义
    parser.add_argument('--visual_prompt_file', type=str, default=None,
                       help='自定义视觉分析 prompt 文件')
    parser.add_argument('--som_prompt_file', type=str, default=None,
                       help='自定义 SoM 匹配 prompt 文件')
    parser.add_argument('--merge_prompt_file', type=str, default=None,
                       help='自定义合并标注 prompt 文件')
    parser.add_argument('--export_prompts', type=str, default=None,
                       help='导出默认 prompt 模板到指定目录（用于修改）')
    
    # MLLM 配置
    parser.add_argument('--mllm_provider', type=str, default='dashscope',
                       choices=['openai', 'anthropic', 'openai-compatible', 'dashscope'],
                       help='MLLM 提供商 (默认: dashscope)')
    parser.add_argument('--mllm_api_key', type=str, default=None,
                       help='MLLM API Key')
    parser.add_argument('--mllm_model', type=str, default=None,
                       help='MLLM 模型名称')
    parser.add_argument('--mllm_base_url', type=str, default=None,
                       help='MLLM Base URL')
    parser.add_argument('--enable_thinking', action='store_true', default=False,
                       help='启用思考模式')
    parser.add_argument('--thinking_budget', type=int, default=4096,
                       help='思考模式的最大 Token 数')
    
    # 聚类参数
    parser.add_argument('--k_neighbors', type=int, default=5,
                       help='KNN 邻居数')
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7],
                       help='层级聚类的 Beta 参数')
    parser.add_argument('--min_cluster_points', type=int, default=100,
                       help='最小簇点数')
    
    # 其他
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='仅生成 prompt 和图像，不调用 MLLM API')
    parser.add_argument('--list_nodes', action='store_true', default=False,
                       help='仅列出层级结构中的所有节点')
    
    args = parser.parse_args()
    
    # 导出 prompt 模板
    if args.export_prompts:
        save_prompt_templates(args.export_prompts)
        return
    
    # 仅列出节点
    if args.list_nodes:
        caption_data = load_caption_json(args.caption_json)
        print("\n层级结构:")
        print_hierarchy_tree(caption_data['hierarchy'])
        print("\n使用 --parent_level <level> --parent_cluster_id <id> 指定要重新标注的父节点")
        return
    
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
    
    # 加载自定义 prompt
    visual_prompt = load_custom_prompt(args.visual_prompt_file, VISUAL_ANALYSIS_PROMPT)
    som_prompt = load_custom_prompt(args.som_prompt_file, SOM_MATCHING_PROMPT)
    merge_prompt = load_custom_prompt(args.merge_prompt_file, MERGE_ANNOTATIONS_PROMPT)
    
    # 创建 MLLM 客户端
    if args.dry_run:
        print("[DRY RUN 模式] 跳过 MLLM API 调用")
        client = None
    else:
        if api_key is None:
            print("错误: 请提供 API Key（通过 --mllm_api_key 或环境变量）")
            print("  或者使用 --dry_run 参数跳过 MLLM 调用")
            sys.exit(1)
        
        client = create_mllm_client(
            args.mllm_provider,
            api_key,
            args.mllm_model,
            args.mllm_base_url,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget
        )
    
    # 执行重新标注
    result = replay_captioning(
        caption_json_path=args.caption_json,
        glb_path=args.glb_path,
        npy_path=args.npy_path,
        feature_path=feature_path,
        parent_level=args.parent_level,
        parent_cluster_id=args.parent_cluster_id,
        output_dir=args.output_dir,
        mllm_client=client,
        visual_prompt_template=visual_prompt,
        som_prompt_template=som_prompt,
        merge_prompt_template=merge_prompt,
        k_neighbors=args.k_neighbors,
        betas=args.betas,
        min_cluster_points=args.min_cluster_points,
        save_images=args.save_images,
        dry_run=args.dry_run
    )
    
    return result


if __name__ == '__main__':
    main()

