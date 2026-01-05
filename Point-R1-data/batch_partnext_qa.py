"""
PartNeXt 批量 QA 生成脚本

支持多进程并发处理，提高批量标注效率。

使用方法：
    # 串行处理（默认）
    python batch_partnext_qa.py \
        --partnext_glb_dir /path/to/glbs \
        --partnext_ann_dir /path/to/annotations \
        --output_dir outputs/partnext_qa

    # 并发处理（推荐）
    python batch_partnext_qa.py \
        --partnext_glb_dir /mnt/data/code/Point-R1/PartNeXt_mesh/glbs \
        --partnext_ann_dir /mnt/data/code/Point-R1/PartNeXt_data \
        --output_dir /mnt/data/code/Point-R1/outputs/partnext_qa_batch_test \
        --max_objects 20 \
        --num_workers 4 \
        --num_views 6 \
        --generate_qa

    # 试跑 20 条
    python batch_partnext_qa.py \
        --partnext_glb_dir /path/to/glbs \
        --partnext_ann_dir /path/to/annotations \
        --output_dir outputs/partnext_qa \
        --max_objects 20 \
        --num_workers 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保可以从同目录 import
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    # PartNeXt 是可选依赖：仓库内路径为 PartNeXt/PartNeXt_lib
    PARTNEXT_ROOT = Path(__file__).resolve().parents[1] / "PartNeXt" / "PartNeXt_lib"
    if PARTNEXT_ROOT.exists() and str(PARTNEXT_ROOT) not in sys.path:
        sys.path.insert(0, str(PARTNEXT_ROOT))
    from partnext import PartNeXtDataset  # type: ignore

    PARTNEXT_AVAILABLE = True
except Exception:
    PARTNEXT_AVAILABLE = False


@dataclass
class ProcessResult:
    object_id: str
    status: str  # "success" | "failed" | "skipped"
    num_nodes: int = 0
    num_qas: int = 0
    output_path: str = ""
    qa_output_path: str = ""
    error: str = ""
    processing_time: float = 0.0


def get_all_object_ids(glb_dir: str, ann_dir: str) -> List[str]:
    """获取所有可用的 object_id 列表"""
    if not PARTNEXT_AVAILABLE:
        raise RuntimeError("PartNeXt 不可用")
    
    dataset = PartNeXtDataset(glb_dir, ann_dir)
    return list(dataset.get_object_ids())


def process_single_object(
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
    part_sample_points: int,
    generate_qa: bool,
    qa_part_nodes: str,
    qa_max_parts: int,
    qa_num_other: int,
    qa_max_points: int,
    qa_max_tokens: int,
    qa_temperature: float,
    seed: int,
    qa_debug_print_prompts: bool,
    qa_debug_print_responses: bool,
    mllm_provider: str,
    mllm_api_key: Optional[str],
    mllm_model: Optional[str],
    mllm_base_url: Optional[str],
    enable_thinking: bool,
    thinking_budget: int,
) -> ProcessResult:
    """
    处理单个物体（在子进程中执行）
    
    注意：每个子进程都需要独立创建 MLLM client，不能跨进程共享。
    """
    start_time = time.time()
    
    try:
        # 在子进程中导入，避免主进程加载过多
        from gt_tree_captioning import process_partnext_captioning
        from hierarchical_captioning import create_mllm_client
        
        # 创建 MLLM client（每个进程独立创建）
        if dry_run:
            client = None
        else:
            api_key = mllm_api_key
            if api_key is None:
                if mllm_provider == "openai":
                    api_key = os.environ.get("OPENAI_API_KEY")
                elif mllm_provider == "anthropic":
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                elif mllm_provider == "dashscope":
                    api_key = os.environ.get("DASHSCOPE_API_KEY")
                else:
                    api_key = os.environ.get("MLLM_API_KEY")
            
            if not api_key:
                return ProcessResult(
                    object_id=object_id,
                    status="failed",
                    error="缺少 API Key",
                    processing_time=time.time() - start_time,
                )
            
            client = create_mllm_client(
                mllm_provider,
                api_key,
                mllm_model,
                mllm_base_url,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
        
        # 为每个物体创建独立输出目录
        object_output_dir = os.path.join(output_dir, object_id)
        qa_output_path = os.path.join(object_output_dir, f"{object_id}_qa.jsonl")
        
        result = process_partnext_captioning(
            glb_dir=glb_dir,
            ann_dir=ann_dir,
            object_id=object_id,
            output_dir=object_output_dir,
            image_size=image_size,
            num_views=num_views,
            view_radius=view_radius,
            traversal=traversal,
            save_images=save_images,
            dry_run=dry_run,
            mllm_client=client,
            part_sample_points=part_sample_points,
            generate_qa=generate_qa,
            qa_output_path=qa_output_path,
            qa_part_nodes=qa_part_nodes,
            qa_max_parts=qa_max_parts,
            qa_num_other=qa_num_other,
            qa_max_points=qa_max_points,
            qa_max_tokens=qa_max_tokens,
            qa_temperature=qa_temperature,
            seed=seed,
            qa_debug_print_prompts=qa_debug_print_prompts,
            qa_debug_print_responses=qa_debug_print_responses,
        )
        
        # 统计 QA 数量
        num_qas = 0
        if os.path.exists(qa_output_path):
            with open(qa_output_path, "r", encoding="utf-8") as f:
                num_qas = sum(1 for _ in f)
        
        # 强制垃圾回收，释放 GPU/内存资源
        gc.collect()
        
        return ProcessResult(
            object_id=object_id,
            status="success",
            num_nodes=result.get("num_nodes", 0),
            num_qas=num_qas,
            output_path=result.get("output_path", ""),
            qa_output_path=qa_output_path if os.path.exists(qa_output_path) else "",
            processing_time=time.time() - start_time,
        )
    
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        return ProcessResult(
            object_id=object_id,
            status="failed",
            error=str(e),
            processing_time=time.time() - start_time,
        )


def process_single_object_wrapper(args: Tuple) -> ProcessResult:
    """包装函数，用于 ProcessPoolExecutor"""
    return process_single_object(*args)


def main():
    parser = argparse.ArgumentParser(description="PartNeXt 批量 QA 生成")
    
    # PartNeXt 数据路径
    parser.add_argument("--partnext_glb_dir", type=str, required=True,
                        help="PartNeXt GLB 根目录（如 data/data）")
    parser.add_argument("--partnext_ann_dir", type=str, required=True,
                        help="PartNeXt 标注目录（load_from_disk 的目录）")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/partnext_qa_batch",
                        help="输出目录")
    
    # 物体选择
    parser.add_argument("--object_ids", type=str, nargs="+", default=None,
                        help="指定要处理的对象 ID 列表（不指定则处理所有）")
    parser.add_argument("--max_objects", type=int, default=None,
                        help="最多处理的物体数量（用于试跑）")
    parser.add_argument("--start_index", type=int, default=0,
                        help="从第几个物体开始处理（用于断点续传）")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="随机打乱物体顺序（用于分布式多机）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 并发控制
    parser.add_argument("--num_workers", type=int, default=5,
                        help="并发处理的进程数量（默认 1，即串行）")
    
    # 渲染参数
    parser.add_argument("--image_size", type=int, default=800,
                        help="渲染分辨率（正方形）")
    parser.add_argument("--num_views", type=int, default=6,
                        help="每个节点使用的视角数（<=6）")
    parser.add_argument("--view_radius", type=float, default=1.2,
                        help="相机半径（建议 1.2~2.0）")
    parser.add_argument("--traversal", type=str, default="bfs", choices=["bfs", "dfs"],
                        help="遍历顺序")
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="保存每个节点的输入图像（批量生成时建议关闭）")
    parser.add_argument("--partnext_part_sample_points", type=int, default=5000,
                        help="每个叶子部件采样点数")
    
    # QA 生成参数
    parser.add_argument("--generate_qa", action="store_true", default=True,
                        help="生成 QA 数据（默认开启）")
    parser.add_argument("--qa_part_nodes", type=str, default="all", choices=["leaf", "all"],
                        help="用于生成部件 QA 的节点集合")
    parser.add_argument("--qa_max_parts", type=int, default=20,
                        help="最多生成多少条部件 QA")
    parser.add_argument("--qa_num_other", type=int, default=6,
                        help="生成多少条其他形式 QA")
    parser.add_argument("--qa_max_points", type=int, default=200000,
                        help="QA 高亮渲染时点云最多使用多少点")
    parser.add_argument("--qa_max_tokens", type=int, default=8192 * 4,
                        help="QA 生成 max_tokens")
    parser.add_argument("--qa_temperature", type=float, default=0.6,
                        help="QA 生成 temperature")
    parser.add_argument("--qa_debug_print_prompts", action="store_true", default=False,
                        help="打印 QA prompt 到 stdout（批量生成时建议关闭）")
    parser.add_argument("--qa_debug_print_responses", action="store_true", default=False,
                        help="打印 QA 模型原始回答到 stdout（用于 debug；多进程时会比较刷屏）")
    
    # 运行模式
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="不调用 MLLM，仅跑渲染与占位输出")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="跳过已处理的对象")
    
    # MLLM 配置
    parser.add_argument("--mllm_provider", type=str, default="dashscope",
                        choices=["openai", "anthropic", "openai-compatible", "dashscope"])
    parser.add_argument("--mllm_api_key", type=str, default=None,
                        help="API Key（或使用环境变量）")
    parser.add_argument("--mllm_model", type=str, default=None,
                        help="模型名（dashscope 默认 qwen3-vl-plus）")
    parser.add_argument("--mllm_base_url", type=str, default=None,
                        help="自定义 base_url（openai-compatible/dashscope）")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="dashscope 思考模式")
    parser.add_argument("--thinking_budget", type=int, default=4096,
                        help="dashscope 思考 token 上限")
    
    args = parser.parse_args()
    
    # 检查 PartNeXt 可用性
    if not PARTNEXT_AVAILABLE:
        print("错误: PartNeXt 不可用。请确保仓库内存在 `PartNeXt/PartNeXt_lib`。")
        sys.exit(1)
    
    # 验证路径
    if not os.path.exists(args.partnext_glb_dir):
        print(f"错误: GLB 目录不存在: {args.partnext_glb_dir}")
        sys.exit(1)
    if not os.path.exists(args.partnext_ann_dir):
        print(f"错误: 标注目录不存在: {args.partnext_ann_dir}")
        sys.exit(1)
    
    # 验证 API Key（非 dry_run 时）
    if not args.dry_run:
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
            print("错误: 缺少 API Key。请用 --mllm_api_key 或设置对应环境变量（或使用 --dry_run）")
            sys.exit(1)
    
    # 获取物体列表
    print(f"正在加载 PartNeXt 数据集...")
    if args.object_ids:
        object_ids = args.object_ids
    else:
        object_ids = get_all_object_ids(args.partnext_glb_dir, args.partnext_ann_dir)
    
    print(f"数据集共有 {len(object_ids)} 个物体")
    
    # 随机打乱
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(object_ids)
    
    # 切片
    object_ids = object_ids[args.start_index:]
    if args.max_objects is not None:
        object_ids = object_ids[:args.max_objects]
    
    print(f"本次处理 {len(object_ids)} 个物体 (start_index={args.start_index}, max_objects={args.max_objects})")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resume: 过滤已处理的对象
    if args.resume:
        filtered_ids = []
        for oid in object_ids:
            qa_file = os.path.join(args.output_dir, oid, f"{oid}_qa.jsonl")
            if os.path.exists(qa_file):
                print(f"[跳过] 已处理: {oid}")
            else:
                filtered_ids.append(oid)
        object_ids = filtered_ids
        print(f"过滤后剩余 {len(object_ids)} 个待处理对象")
    
    if not object_ids:
        print("没有需要处理的物体。")
        sys.exit(0)
    
    # 构建参数元组列表
    task_args = [
        (
            args.partnext_glb_dir,
            args.partnext_ann_dir,
            oid,
            args.output_dir,
            args.image_size,
            args.num_views,
            args.view_radius,
            args.traversal,
            args.save_images,
            args.dry_run,
            args.partnext_part_sample_points,
            args.generate_qa,
            args.qa_part_nodes,
            args.qa_max_parts,
            args.qa_num_other,
            args.qa_max_points,
            args.qa_max_tokens,
            args.qa_temperature,
            args.seed,
            args.qa_debug_print_prompts,
            args.qa_debug_print_responses,
            args.mllm_provider,
            args.mllm_api_key,
            args.mllm_model,
            args.mllm_base_url,
            args.enable_thinking,
            args.thinking_budget,
        )
        for oid in object_ids
    ]
    
    # 处理结果
    results: List[ProcessResult] = []
    failed: List[ProcessResult] = []
    
    start_time = time.time()
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("提示: 安装 tqdm 可获得进度条 (pip install tqdm)")
    
    if args.num_workers <= 1:
        # 串行处理
        print(f"\n串行处理 {len(task_args)} 个对象...")
        iterator = tqdm(task_args, desc="处理对象") if use_tqdm else task_args
        for task in iterator:
            result = process_single_object_wrapper(task)
            if result.status == "success":
                results.append(result)
                print(f"[成功] {result.object_id}: nodes={result.num_nodes}, qas={result.num_qas}, time={result.processing_time:.1f}s")
            else:
                failed.append(result)
                print(f"[失败] {result.object_id}: {result.error}")
    else:
        # 多进程并发处理
        print(f"\n并发处理 {len(task_args)} 个对象 (workers={args.num_workers})...")
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_object_wrapper, task): task[2] for task in task_args}
            
            if use_tqdm:
                pbar = tqdm(total=len(futures), desc="处理对象")
            
            for future in as_completed(futures):
                object_id = futures[future]
                try:
                    result = future.result()
                    if result.status == "success":
                        results.append(result)
                        print(f"[成功] {result.object_id}: nodes={result.num_nodes}, qas={result.num_qas}, time={result.processing_time:.1f}s")
                    else:
                        failed.append(result)
                        print(f"[失败] {result.object_id}: {result.error}")
                except Exception as e:
                    failed.append(ProcessResult(
                        object_id=object_id,
                        status="failed",
                        error=f"进程异常: {e}",
                    ))
                    print(f"[异常] {object_id}: {e}")
                
                if use_tqdm:
                    pbar.update(1)
            
            if use_tqdm:
                pbar.close()
    
    total_time = time.time() - start_time
    
    # 保存汇总结果
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "total_objects": len(task_args),
        "success": len(results),
        "failed": len(failed),
        "total_time_seconds": total_time,
        "avg_time_per_object": total_time / len(task_args) if task_args else 0,
        "total_qas": sum(r.num_qas for r in results),
        "results": [asdict(r) for r in results],
        "failures": [asdict(r) for r in failed],
    }
    
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 合并所有 QA 到一个文件
    merged_qa_path = os.path.join(args.output_dir, "merged_qa.jsonl")
    with open(merged_qa_path, "w", encoding="utf-8") as fout:
        for r in results:
            if r.qa_output_path and os.path.exists(r.qa_output_path):
                with open(r.qa_output_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
    
    print(f"\n{'=' * 60}")
    print(f"批量处理完成!")
    print(f"  成功: {len(results)}")
    print(f"  失败: {len(failed)}")
    print(f"  总 QA 数: {summary['total_qas']}")
    print(f"  总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"  平均每物体: {summary['avg_time_per_object']:.1f}s")
    print(f"  汇总保存至: {summary_path}")
    print(f"  合并 QA 保存至: {merged_qa_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

