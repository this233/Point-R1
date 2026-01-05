"""
批量层级化标注脚本

支持多物体并发处理，提高批量标注效率。
支持断点续传，可随时中断和重启。

特性：
    - 断点续传：默认启用，随时可以 Ctrl+C 中断，下次继续
    - 进度追踪：实时保存进度到 progress.json
    - 失败重试：支持重试失败的对象，带最大重试次数限制
    - 并发处理：支持多线程并发处理

使用方法：
    # 基础用法（串行处理，自动断点续传）
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions
    
    # 并发处理（推荐，4个worker）
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions \
        --num_workers 4
    
    # 查看当前进度
    python batch_captioning.py \
        --output_dir outputs/captions \
        --show_progress
    
    # 重试之前失败的对象
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions \
        --retry_failed
    
    # 重置进度，重新开始
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions \
        --reset_progress
    
    # 禁用断点续传，重新处理所有对象
    python batch_captioning.py \
        --input_dir example_material \
        --output_dir outputs/captions \
        --no_resume


# 正常运行（自动断点续传）
python batch_captioning.py --input_dir /path/to/data --output_dir outputs/captions --num_workers 4

# 查看进度
python batch_captioning.py --output_dir outputs/captions --show_progress

# 重试失败的对象
python batch_captioning.py --input_dir /path/to/data --output_dir outputs/captions --retry_failed

# 完全重新开始
python batch_captioning.py --input_dir /path/to/data --output_dir outputs/captions --reset_progress
"""

import os
import sys
import argparse
import json
import fcntl
import signal
import atexit
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from tqdm import tqdm

from hierarchical_captioning import (
    create_mllm_client,
    process_hierarchical_captioning
)


class ProgressTracker:
    """进度追踪器，支持断点续传"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, 'progress.json')
        self.lock = Lock()
        self.progress = self._load_progress()
        self._shutdown_event = Event()
        
        # 注册退出处理
        atexit.register(self._save_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_progress(self) -> dict:
        """加载进度文件"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    data = json.load(f)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 进度文件损坏，将创建新文件: {e}")
        
        return {
            'completed': {},      # object_id -> {result_info}
            'failed': {},         # object_id -> {error, timestamp, attempts}
            'in_progress': [],    # 当前正在处理的对象
            'started_at': None,
            'last_updated': None
        }
    
    def _save_progress(self):
        """保存进度文件（线程安全）"""
        with self.lock:
            self.progress['last_updated'] = datetime.now().isoformat()
            
            # 使用临时文件 + 重命名确保原子性
            temp_file = self.progress_file + '.tmp'
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(self.progress, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                os.rename(temp_file, self.progress_file)
            except Exception as e:
                print(f"警告: 保存进度失败: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n收到信号 {signum}，正在保存进度...")
        self._shutdown_event.set()
        self._save_progress()
        print("进度已保存，可以安全退出。")
        sys.exit(0)
    
    def _save_on_exit(self):
        """退出时保存进度"""
        if not self._shutdown_event.is_set():
            self._save_progress()
    
    def should_shutdown(self) -> bool:
        """检查是否应该停止处理"""
        return self._shutdown_event.is_set()
    
    def start_session(self):
        """开始新的处理会话"""
        with self.lock:
            if self.progress['started_at'] is None:
                self.progress['started_at'] = datetime.now().isoformat()
            # 清理上次未完成的 in_progress 状态
            self.progress['in_progress'] = []
        self._save_progress()
    
    def get_completed_ids(self) -> set:
        """获取已完成的对象 ID"""
        with self.lock:
            return set(self.progress['completed'].keys())
    
    def get_failed_ids(self) -> set:
        """获取失败的对象 ID"""
        with self.lock:
            return set(self.progress['failed'].keys())
    
    def mark_in_progress(self, object_id: str):
        """标记对象为处理中"""
        with self.lock:
            if object_id not in self.progress['in_progress']:
                self.progress['in_progress'].append(object_id)
        self._save_progress()
    
    def mark_completed(self, object_id: str, result: dict):
        """标记对象为已完成"""
        with self.lock:
            self.progress['completed'][object_id] = {
                'timestamp': datetime.now().isoformat(),
                **result
            }
            # 从失败列表和进行中列表移除
            self.progress['failed'].pop(object_id, None)
            if object_id in self.progress['in_progress']:
                self.progress['in_progress'].remove(object_id)
        self._save_progress()
    
    def mark_failed(self, object_id: str, error: str):
        """标记对象为失败"""
        with self.lock:
            attempts = self.progress['failed'].get(object_id, {}).get('attempts', 0) + 1
            self.progress['failed'][object_id] = {
                'error': error,
                'timestamp': datetime.now().isoformat(),
                'attempts': attempts
            }
            if object_id in self.progress['in_progress']:
                self.progress['in_progress'].remove(object_id)
        self._save_progress()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                'completed': len(self.progress['completed']),
                'failed': len(self.progress['failed']),
                'in_progress': len(self.progress['in_progress'])
            }
    
    def get_summary(self) -> dict:
        """获取完整摘要"""
        with self.lock:
            return {
                'started_at': self.progress['started_at'],
                'last_updated': self.progress['last_updated'],
                'completed_count': len(self.progress['completed']),
                'failed_count': len(self.progress['failed']),
                'completed': self.progress['completed'],
                'failed': self.progress['failed']
            }


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
    parser = argparse.ArgumentParser(description='批量层级化 3D 物体标注（支持断点续传）')
    
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
    
    # 断点续传（默认启用）
    parser.add_argument('--no_resume', action='store_true', default=False,
                       help='禁用断点续传，重新处理所有对象')
    parser.add_argument('--retry_failed', action='store_true', default=False,
                       help='重试之前失败的对象')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='单个对象最大重试次数（默认 3）')
    parser.add_argument('--show_progress', action='store_true', default=False,
                       help='显示当前进度统计后退出')
    parser.add_argument('--reset_progress', action='store_true', default=False,
                       help='重置进度文件，重新开始')
    
    # 并发控制
    parser.add_argument('--num_workers', type=int, default=1,
                       help='并发处理的物体数量（默认 1，即串行）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化进度追踪器
    progress_tracker = ProgressTracker(args.output_dir)
    
    # 重置进度
    if args.reset_progress:
        if os.path.exists(progress_tracker.progress_file):
            os.remove(progress_tracker.progress_file)
            print(f"已删除进度文件: {progress_tracker.progress_file}")
            # 重新初始化
            progress_tracker = ProgressTracker(args.output_dir)
        else:
            print("进度文件不存在，无需重置")
    
    # 显示进度统计
    if args.show_progress:
        summary = progress_tracker.get_summary()
        print(f"\n{'='*60}")
        print("当前进度统计:")
        print(f"  开始时间: {summary['started_at'] or '未开始'}")
        print(f"  最后更新: {summary['last_updated'] or '无'}")
        print(f"  已完成: {summary['completed_count']}")
        print(f"  失败: {summary['failed_count']}")
        
        if summary['failed_count'] > 0:
            print(f"\n失败的对象:")
            for obj_id, info in summary['failed'].items():
                error_preview = info['error'][:50] if len(info['error']) > 50 else info['error']
                print(f"    {obj_id}: {error_preview}... (尝试 {info['attempts']} 次)")
        print(f"{'='*60}\n")
        sys.exit(0)
    
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
    
    total_found = len(matched_files)
    print(f"\n找到 {total_found} 个对象")
    
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
    
    # 断点续传：过滤已完成的对象（默认启用）
    completed_ids = progress_tracker.get_completed_ids()
    failed_ids = progress_tracker.get_failed_ids()
    
    if args.no_resume:
        # 禁用断点续传，重新处理所有对象
        print("  [断点续传已禁用] 将重新处理所有对象")
        if completed_ids:
            print(f"  警告: 将覆盖 {len(completed_ids)} 个已完成的对象")
    else:
        # 默认启用断点续传
        filtered_files = []
        skipped_completed = 0
        skipped_failed = 0
        skipped_max_retries = 0
        will_retry = 0
        
        for item in matched_files:
            object_id = item['object_id']
            
            # 已完成的跳过
            if object_id in completed_ids:
                skipped_completed += 1
                continue
            
            # 检查失败的对象
            if object_id in failed_ids:
                failed_info = progress_tracker.progress['failed'].get(object_id, {})
                attempts = failed_info.get('attempts', 0)
                
                if args.retry_failed:
                    # 检查是否超过最大重试次数
                    if attempts >= args.max_retries:
                        skipped_max_retries += 1
                        continue
                    will_retry += 1
                    filtered_files.append(item)
                else:
                    # 默认不自动重试失败的
                    skipped_failed += 1
                    continue
            else:
                filtered_files.append(item)
        
        matched_files = filtered_files
        
        print(f"  已完成（跳过）: {skipped_completed}")
        if args.retry_failed:
            print(f"  将重试: {will_retry}")
            if skipped_max_retries > 0:
                print(f"  超过最大重试次数（跳过）: {skipped_max_retries}")
        elif skipped_failed > 0:
            print(f"  之前失败（跳过，使用 --retry_failed 重试）: {skipped_failed}")
        print(f"  待处理: {len(matched_files)}")
    
    if not matched_files:
        print("\n没有待处理的对象")
        stats = progress_tracker.get_stats()
        print(f"  已完成: {stats['completed']}, 失败: {stats['failed']}")
        sys.exit(0)
    
    # 开始处理会话
    progress_tracker.start_session()
    
    def process_single_object(item: dict) -> dict:
        """处理单个物体（线程安全，带进度追踪）"""
        object_id = item['object_id']
        object_output_dir = os.path.join(args.output_dir, object_id)
        
        # 检查是否应该停止
        if progress_tracker.should_shutdown():
            return {
                'object_id': object_id,
                'status': 'skipped',
                'error': '用户中断'
            }
        
        # 标记为处理中
        progress_tracker.mark_in_progress(object_id)
        
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
            
            result_info = {
                'total_clusters': result.total_clusters,
                'total_levels': result.total_levels,
                'processing_time': result.processing_time
            }
            
            # 立即保存进度
            progress_tracker.mark_completed(object_id, result_info)
            
            return {
                'object_id': object_id,
                'status': 'success',
                **result_info
            }
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            
            # 立即保存失败状态
            progress_tracker.mark_failed(object_id, error_msg)
            
            return {
                'object_id': object_id,
                'status': 'failed',
                'error': error_msg
            }
    
    # 处理结果统计（本次会话）
    session_success = 0
    session_failed = 0
    
    # 根据 num_workers 决定串行还是并发
    if args.num_workers <= 1:
        # 串行处理
        print(f"\n串行处理 {len(matched_files)} 个对象...")
        print("提示: 可随时按 Ctrl+C 中断，进度会自动保存\n")
        
        for item in tqdm(matched_files, desc="处理对象"):
            if progress_tracker.should_shutdown():
                print("\n用户中断，停止处理...")
                break
            
            result = process_single_object(item)
            if result['status'] == 'success':
                session_success += 1
            elif result['status'] == 'failed':
                session_failed += 1
    else:
        # 并发处理
        print(f"\n并发处理 {len(matched_files)} 个对象 (workers={args.num_workers})...")
        print("提示: 可随时按 Ctrl+C 中断，进度会自动保存\n")
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_object, item): item 
                      for item in matched_files}
            
            try:
                with tqdm(total=len(futures), desc="处理对象") as pbar:
                    for future in as_completed(futures):
                        if progress_tracker.should_shutdown():
                            # 取消未开始的任务
                            for f in futures:
                                f.cancel()
                            break
                        
                        result = future.result()
                        if result['status'] == 'success':
                            session_success += 1
                        elif result['status'] == 'failed':
                            session_failed += 1
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\n用户中断，正在等待当前任务完成...")
                executor.shutdown(wait=False, cancel_futures=True)
    
    # 获取总体统计
    stats = progress_tracker.get_stats()
    
    # 保存汇总结果
    summary = progress_tracker.get_summary()
    summary['session'] = {
        'timestamp': datetime.now().isoformat(),
        'processed': session_success + session_failed,
        'success': session_success,
        'failed': session_failed
    }
    
    summary_path = os.path.join(args.output_dir, 'batch_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"\n本次会话:")
    print(f"  成功: {session_success}")
    print(f"  失败: {session_failed}")
    print(f"\n总体进度:")
    print(f"  已完成: {stats['completed']}")
    print(f"  失败: {stats['failed']}")
    
    if stats['failed'] > 0:
        print(f"\n提示: 使用 --retry_failed 重试失败的对象")
    
    print(f"\n进度文件: {progress_tracker.progress_file}")
    print(f"汇总文件: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()



