"""
批量层级化标注脚本

支持多物体并发处理，提高批量标注效率。

使用方法：
    # 串行处理（默认）
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions
    
    # 并发处理（推荐）
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions \
        --num_workers 4
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from hierarchical_captioning import (
    create_mllm_client,
    process_hierarchical_captioning
)


def find_matching_files(input_dir: str) -> list:
    """查找匹配的 GLB 和 NPY 文件"""
    input_path = Path(input_dir)
    
    glb_dir = input_path / 'glbs'
    npy_dir = input_path / 'npys'
    feature_dir = input_path / 'dino_features'
    
    if not glb_dir.exists():
        print(f"错误: GLB 目录不存在: {glb_dir}")
        return []
    
    if not npy_dir.exists():
        print(f"错误: NPY 目录不存在: {npy_dir}")
        return []
    
    # 查找所有 GLB 文件
    glb_files = list(glb_dir.glob('*.glb'))
    
    matched = []
    for glb_file in glb_files:
        object_id = glb_file.stem
        npy_file = npy_dir / f"{object_id}_8192.npy"
        feature_file = feature_dir / f"{object_id}_features.npy"
        
        if not npy_file.exists():
            print(f"警告: 找不到点云文件 {npy_file}")
            continue
        
        if not feature_file.exists():
            print(f"警告: 找不到特征文件 {feature_file}")
            print(f"  请先运行: python extract_dino_features.py --object_id {object_id}")
            continue
        
        matched.append({
            'object_id': object_id,
            'glb_path': str(glb_file),
            'npy_path': str(npy_file),
            'feature_path': str(feature_file)
        })
    
    return matched


def main():
    parser = argparse.ArgumentParser(description='批量层级化 3D 物体标注')
    
    # 输入输出
    parser.add_argument('--input_dir', type=str, default='example_material',
                       help='输入目录（包含 glbs/, npys/, dino_features/ 子目录）')
    parser.add_argument('--output_dir', type=str, default='outputs/captions',
                       help='输出目录')
    parser.add_argument('--object_ids', type=str, nargs='+', default=None,
                       help='指定要处理的对象 ID 列表（不指定则处理所有）')
    
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
    parser.add_argument('--k_neighbors', type=int, default=5)
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7])
    
    # 其他参数
    parser.add_argument('--max_depth', type=int, default=4,
                       help='最大层级深度')
    parser.add_argument('--min_cluster_points', type=int, default=100,
                       help='最小簇点数')
    parser.add_argument('--save_images', action='store_true', default=True,
                       help='是否保存中间图像')
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='仅执行聚类和渲染，不调用 MLLM API（用于调试）')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='跳过已处理的对象')
    
    # 并发控制
    parser.add_argument('--num_workers', type=int, default=1,
                       help='并发处理的物体数量（默认 1，即串行）')
    
    args = parser.parse_args()
    
    # 获取 API Key
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
    
    # dry_run 模式下不需要 API key
    if args.dry_run:
        print("[DRY RUN 模式] 跳过 MLLM API 调用，仅执行聚类和渲染")
        client = None
    elif api_key is None:
        print("错误: 请提供 API Key（通过 --mllm_api_key 或环境变量）")
        print("  DashScope: DASHSCOPE_API_KEY")
        print("  OpenAI: OPENAI_API_KEY")
        print("  Anthropic: ANTHROPIC_API_KEY")
        print("  或者使用 --dry_run 参数跳过 MLLM 调用")
        sys.exit(1)
    
    # 查找匹配文件
    matched_files = find_matching_files(args.input_dir)
    
    if not matched_files:
        print("没有找到可处理的文件")
        sys.exit(1)
    
    # 过滤指定的对象
    if args.object_ids:
        matched_files = [f for f in matched_files if f['object_id'] in args.object_ids]
    
    print(f"\n找到 {len(matched_files)} 个待处理对象")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建 MLLM 客户端
    if not args.dry_run:
        client = create_mllm_client(
            args.mllm_provider,
            api_key,
            args.mllm_model,
            args.mllm_base_url,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget
        )
    
    # 处理结果
    results = []
    failed = []
    results_lock = Lock()
    
    # 过滤已处理的对象
    if args.resume:
        filtered_files = []
        for item in matched_files:
            output_json = os.path.join(args.output_dir, item['object_id'], f"{item['object_id']}_caption.json")
            if os.path.exists(output_json):
                print(f"跳过已处理: {item['object_id']}")
            else:
                filtered_files.append(item)
        matched_files = filtered_files
        print(f"过滤后剩余 {len(matched_files)} 个待处理对象")
    
    def process_single_object(item: dict) -> dict:
        """处理单个物体（线程安全）"""
        object_id = item['object_id']
        object_output_dir = os.path.join(args.output_dir, object_id)
        
        try:
            result = process_hierarchical_captioning(
                glb_path=item['glb_path'],
                npy_path=item['npy_path'],
                feature_path=item['feature_path'],
                output_dir=object_output_dir,
                mllm_client=client,
                k_neighbors=args.k_neighbors,
                betas=args.betas,
                max_depth=args.max_depth,
                min_cluster_points=args.min_cluster_points,
                save_images=args.save_images,
                dry_run=args.dry_run
            )
            
            return {
                'object_id': object_id,
                'status': 'success',
                'total_clusters': result.total_clusters,
                'total_levels': result.total_levels,
                'processing_time': result.processing_time
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'object_id': object_id,
                'status': 'failed',
                'error': str(e)
            }
    
    # 根据 num_workers 决定串行还是并发
    if args.num_workers <= 1:
        # 串行处理
        print(f"\n串行处理 {len(matched_files)} 个对象...")
        for item in tqdm(matched_files, desc="处理对象"):
            result = process_single_object(item)
            if result['status'] == 'success':
                results.append(result)
            else:
                failed.append(result)
    else:
        # 并发处理
        print(f"\n并发处理 {len(matched_files)} 个对象 (workers={args.num_workers})...")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_object, item): item 
                      for item in matched_files}
            
            with tqdm(total=len(futures), desc="处理对象") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result['status'] == 'success':
                        with results_lock:
                            results.append(result)
                    else:
                        with results_lock:
                            failed.append(result)
                    pbar.update(1)
    
    # 保存汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_objects': len(matched_files),
        'success': len(results),
        'failed': len(failed),
        'results': results,
        'failures': failed
    }
    
    summary_path = os.path.join(args.output_dir, 'batch_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"  成功: {len(results)}")
    print(f"  失败: {len(failed)}")
    print(f"  汇总保存至: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()



