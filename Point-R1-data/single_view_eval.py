"""
单视角评估脚本：把每个候选视角分别送给 MLLM，让你统计哪些视角能分对左右部件。

设计目标：
1) 尽量复用 Point-R1-data/hierarchical_captioning.py 里的已有函数（通过 import）
2) 复用同款 prompt 模板（例如 step1b_glb_overlay_naming），但每次只喂 1 张图
3) 输出每个视角的渲染图 + MLLM 原始响应 + 结构化解析结果（CSV/JSON）

典型用法（以 L2_C3 为例，partition=4 -> 42 个候选视角）：
python Point-R1-data/single_view_eval.py \
  --glb_path example_material/glbs/e85ebb729b02402bbe3b917e1196f8d3.glb \
  --npy_path example_material/npys/e85ebb729b02402bbe3b917e1196f8d3_8192.npy \
  --output_dir outputs/captions \
  --eval_level 2 --eval_cluster 3 \
  --view_partition 4 \
  --prompt_name step1b_glb_overlay_naming \
  --expected_green left --expected_yellow right
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import numpy as np
from PIL import Image

# 复用主脚本能力
import hierarchical_captioning as hc


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里尽量提取一个 JSON 对象。"""
    if not text:
        return None
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        return None
    return None


def _normalize_lr(name: str) -> Optional[str]:
    """
    把模型输出的部件名称归一化为 left/right（物体自身左右）。
    返回: "left" | "right" | None
    """
    if not name:
        return None
    s = str(name).strip().lower()
    # 常见中文写法（尽量宽松）
    left_keys = ["左前肢", "左前腿", "左前爪", "左前臂", "左前足", "左前"]
    right_keys = ["右前肢", "右前腿", "右前爪", "右前臂", "右前足", "右前"]
    for k in left_keys:
        if k in s:
            return "left"
    for k in right_keys:
        if k in s:
            return "right"
    return None


def _guess_part_names_map(parsed: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    将 {part_names:[{som_id,...},...]} 转成 som_id->entry。
    """
    out: Dict[int, Dict[str, Any]] = {}
    if not parsed:
        return out
    part_names = parsed.get("part_names") or []
    if not isinstance(part_names, list):
        return out
    for p in part_names:
        if not isinstance(p, dict):
            continue
        sid = p.get("som_id")
        try:
            sid_i = int(sid)
        except Exception:
            continue
        out[sid_i] = p
    return out


def _compute_az_el(center: np.ndarray, eye: np.ndarray) -> Tuple[float, float, float]:
    vec = eye - center
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return 0.0, 0.0, 0.0
    elevation = float(np.degrees(np.arcsin(vec[1] / dist)))
    azimuth = float(np.degrees(np.arctan2(vec[2], vec[0])))
    if azimuth < 0:
        azimuth += 360.0
    return azimuth, elevation, dist


def _build_single_view_messages(
    client: hc.MLLMClient,
    prompt_text: str,
    image: np.ndarray,
    view_direction: str,
) -> List[Dict[str, Any]]:
    img_b64 = client.encode_image(image)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    content.append({"type": "text", "text": f"[视角 1: {view_direction}]"})
    return [{"role": "user", "content": content}]


def main() -> int:
    parser = argparse.ArgumentParser(description="单视角评估：逐视角喂给MLLM，统计哪些视角能分对左右前肢")

    # 输入
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--feature_path", type=str, default=None, help="不填则按主脚本规则自动推断")
    parser.add_argument("--output_dir", type=str, default="outputs/captions")

    # 评估节点（对应日志文件名 L{level}_C{cluster}_...）
    parser.add_argument("--eval_level", type=int, required=True)
    parser.add_argument("--eval_cluster", type=int, required=True)

    # 视角参数
    parser.add_argument("--view_partition", type=int, default=4, help="partition=4 -> 42个候选视角")
    parser.add_argument("--image_size", type=int, default=800)

    # prompt 选择：复用 prompts/*.txt
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="step1b_glb_overlay_naming",
        help="复用 Point-R1-data/prompts/{name}.txt",
    )

    # 只用哪个图喂给模型：overlay / som / combined(step1c)
    parser.add_argument(
        "--input_image_type",
        type=str,
        choices=["overlay", "som", "step1c_combined"],
        default="overlay",
    )

    # MLLM 配置（复用主脚本 provider/client）
    parser.add_argument("--mllm_provider", type=str, default="dashscope",
                        choices=["openai", "anthropic", "openai-compatible", "dashscope"])
    # 安全：不要在代码里写死 key；请用 --mllm_api_key 或环境变量
    parser.add_argument("--mllm_api_key", type=str, default=None)
    parser.add_argument("--mllm_model", type=str, default=None)
    parser.add_argument("--mllm_base_url", type=str, default=None)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--thinking_budget", type=int, default=4096)

    # 并发与采样
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--dry_run", action="store_true", default=False)

    # 期望答案（用于判对错）
    parser.add_argument("--expected_green", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--expected_yellow", type=str, default="right", choices=["left", "right"])

    # 聚类参数（与主脚本一致，保持可复现）
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.2, 0.35, 0.5])

    # Open3D 在部分版本里退出析构会崩（FilamentResourceManager DestroyResource）。
    # 结果已落盘但进程会 core dump。默认用硬退出规避；如需正常退出（可能仍会崩），加 --no_hard_exit
    parser.add_argument("--no_hard_exit", action="store_true", default=False)

    args = parser.parse_args()

    # 自动推断 feature_path（复用主脚本规则）
    feature_path = args.feature_path
    if feature_path is None:
        object_id = Path(args.npy_path).stem.replace("_8192", "")
        feature_dir = Path(args.npy_path).parent.parent / "dino_features"
        feature_path = str(feature_dir / f"{object_id}_features.npy")

    # 输出目录
    run_dir = os.path.join(
        args.output_dir,
        "single_view_eval",
        f"L{args.eval_level}_C{args.eval_cluster}",
        f"partition{args.view_partition}_{args.input_image_type}",
    )
    images_dir = os.path.join(run_dir, "images")
    responses_dir = os.path.join(run_dir, "responses")
    _safe_mkdir(images_dir)
    _safe_mkdir(responses_dir)

    # client
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
            raise RuntimeError("缺少 API key：请用 --mllm_api_key 或设置对应环境变量")
        client = hc.create_mllm_client(
            args.mllm_provider,
            api_key,
            args.mllm_model,
            args.mllm_base_url,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )

    # 加载数据与聚类（复用主脚本）
    points, clustering_results, model, features = hc.load_and_prepare_data(
        args.glb_path,
        args.npy_path,
        feature_path,
        args.k_neighbors,
        args.betas,
    )

    # 找到 child_level_idx & child_ids（复用主脚本逻辑）
    child_level_idx, child_ids_np, msg = hc.find_children_group(
        clustering_results, args.eval_level, args.eval_cluster
    )
    if child_ids_np is None or child_level_idx is None:
        raise RuntimeError(f"无法找到子节点: {msg}")

    child_labels = clustering_results[child_level_idx]
    child_ids: List[int] = [int(x) for x in list(child_ids_np)]

    # 与 generate_sibling_views 保持一致：按点数排序，决定 som_id_map & 颜色顺序
    child_point_counts = [(cid, int(np.sum(child_labels == cid))) for cid in child_ids]
    child_point_counts.sort(key=lambda x: x[1], reverse=True)
    child_ids = [cid for cid, _ in child_point_counts]
    som_id_map = {cid: i + 1 for i, cid in enumerate(child_ids)}

    DISTINCT_COLORS = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
        [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 255, 128], [255, 0, 128],
        [128, 255, 0], [0, 128, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128, 128, 0], [128, 0, 128], [0, 128, 128], [165, 42, 42], [255, 215, 0],
    ]
    color_map = {cid: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, cid in enumerate(child_ids)}

    # 期望判定：根据颜色名确定“绿色/黄色”的 som_id
    green_sid = None
    yellow_sid = None
    for cid in child_ids:
        sid = som_id_map[cid]
        cname = hc.get_color_name(color_map[cid])
        if cname == "绿色":
            green_sid = sid
        if cname == "黄色":
            yellow_sid = sid
    if green_sid is None or yellow_sid is None:
        raise RuntimeError("当前子簇数量/配色不足，无法定位 绿色/黄色 对应的 som_id（需要至少4个部件）")

    # 全局/父级文本：为了最大化复用 log，这里直接用主脚本的 prompt 模板填充
    # 注意：如果你希望严格复用某一次日志里的 global_caption/parent_caption，可改成从 *_interaction.json 或 *_caption.json 读取。
    global_name = "未知物体"
    global_caption = ""
    parent_name = f"Level{args.eval_level}_Cluster{args.eval_cluster}"
    parent_caption = ""
    try:
        # 尝试从 outputs/*_caption.json 找到全局名/描述（如果存在）
        cap_json = Path(args.output_dir) / f"{Path(args.glb_path).stem}_caption.json"
        if cap_json.exists():
            d = json.loads(cap_json.read_text(encoding="utf-8"))
            global_name = d.get("global_name", global_name)
            global_caption = d.get("global_caption", global_caption)
    except Exception:
        pass

    color_mapping_str = hc.get_color_mapping_str(child_ids, som_id_map, color_map)
    prompt_template = hc.get_prompt(args.prompt_name)
    try:
        prompt_text = prompt_template.format(
            global_name=global_name,
            global_caption=global_caption,
            parent_name=parent_name,
            parent_caption=parent_caption,
            color_mapping=color_mapping_str,
            # step1c 模板里可能会用到下面两个字段；单视角评估通常不走 step1c，但这里给默认占位
            pointcloud_naming_result="（单视角评估：无多轮点云结果）",
            glb_overlay_naming_result="（单视角评估：无多轮叠加蒙版结果）",
        )
    except KeyError:
        # 有些模板不需要所有字段
        prompt_text = prompt_template.format(
            global_name=global_name,
            global_caption=global_caption,
            parent_name=parent_name,
            parent_caption=parent_caption,
            color_mapping=color_mapping_str,
        )

    # 构建组点云
    group_indices: List[int] = []
    group_point_child_ids: List[int] = []
    for cid in child_ids:
        idxs = np.where(child_labels == cid)[0]
        group_indices.extend(idxs.tolist())
        group_point_child_ids.extend([cid] * len(idxs))
    group_points = points[group_indices]
    group_features = features[group_indices]
    group_point_child_ids = np.asarray(group_point_child_ids)
    group_center = np.mean(group_points, axis=0)
    child_masks_in_group = {cid: (group_point_child_ids == cid) for cid in child_ids}

    # 特征显著性（复用主脚本逻辑）
    feat_center = np.mean(group_features, axis=0)
    feat_dists = np.linalg.norm(group_features - feat_center, axis=1)
    if feat_dists.max() > feat_dists.min() + 1e-6:
        feat_weights = (feat_dists - feat_dists.min()) / (feat_dists.max() - feat_dists.min())
        feat_weights = 0.5 + (feat_weights * 2.5)
    else:
        feat_weights = np.ones_like(feat_dists)

    # 渲染器（一个用于快速评估距离/深度，另一个用于输出高清图）
    renderer_fast = None
    renderer_final = None
    renderer_fast = hc.Open3DRenderer(width=256, height=256)
    renderer_fast.setup()
    renderer_fast.upload_model(model)

    renderer_final = hc.Open3DRenderer(width=args.image_size, height=args.image_size)
    renderer_final.setup()
    renderer_final.upload_model(model)

    fov = 60.0
    cam_params_fast = {
        "intrinsic": {
            "width": 256, "height": 256,
            "fx": 256 / (2.0 * np.tan(np.radians(fov) / 2.0)),
            "fy": 256 / (2.0 * np.tan(np.radians(fov) / 2.0)),
            "cx": 128.0, "cy": 128.0,
            "fov": fov,
        }
    }
    intrinsic_fast = hc.create_camera_intrinsic_from_params(cam_params_fast)

    cam_params_final = {
        "intrinsic": {
            "width": args.image_size, "height": args.image_size,
            "fx": args.image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
            "fy": args.image_size / (2.0 * np.tan(np.radians(fov) / 2.0)),
            "cx": args.image_size / 2.0, "cy": args.image_size / 2.0,
            "fov": fov,
        }
    }
    intrinsic_final = hc.create_camera_intrinsic_from_params(cam_params_final)

    # 候选视角
    view_points = hc.sample_view_points(radius=1.0, partition=args.view_partition)
    print(f"[INFO] partition={args.view_partition}, 候选视角数={len(view_points)}")

    # 预计算每个视角的 eye/dist/方位信息 + 渲染输入图像（避免把渲染放进并发请求里，减少GPU/渲染器冲突）
    view_items: List[Dict[str, Any]] = []
    for i, vp in enumerate(view_points):
        direction = vp / np.linalg.norm(vp)
        dist = hc.optimize_distance_for_cluster(
            renderer_fast, group_points, direction, group_center,
            intrinsic_fast, 256, target_occupancy=0.6, min_dist_threshold=0.5
        )
        eye = group_center + direction * dist
        az, el, dist2 = _compute_az_el(group_center, eye)
        view_direction = hc.get_view_direction_description(az, el)

        # 高清渲染 clean + 深度
        clean_img, depth_map = renderer_final.render_view(eye, center=group_center, return_depth=True)

        # overlay / som / step1c_combined
        overlay_img = hc.create_som_overlay_image(
            clean_img, points, child_labels, child_ids, color_map,
            eye, group_center, depth_map, intrinsic_final,
            image_size=args.image_size, alpha=0.3
        )
        som_img = None
        if args.input_image_type in ("som", "step1c_combined"):
            som_img = hc.render_pointcloud_som_image(
                points, child_labels, child_ids, color_map,
                eye, group_center,
                image_size=args.image_size,
                point_size=3.0, dim_factor=0.25, distance=dist
            )

        if args.input_image_type == "overlay":
            input_img = overlay_img
        elif args.input_image_type == "som":
            if som_img is None:
                raise RuntimeError("当前环境无法渲染点云SoM图（OPEN3D不可用），请改用 --input_image_type overlay")
            input_img = som_img
        else:  # step1c_combined
            if som_img is None:
                raise RuntimeError("当前环境无法渲染点云SoM图（OPEN3D不可用），请改用 --input_image_type overlay")
            input_img = np.concatenate([som_img, overlay_img], axis=1)

        # 保存输入图
        img_path = os.path.join(images_dir, f"view_{i:03d}_az{az:.1f}_el{el:.1f}.png")
        Image.fromarray(input_img).save(img_path)

        view_items.append({
            "view_idx": i,
            "eye": eye,
            "dist": float(dist2),
            "azimuth": az,
            "elevation": el,
            "view_direction": view_direction,
            "img_path": img_path,
            "input_img": input_img,  # 后面会 base64 编码发送
        })

    # 逐视角调用
    results: List[Dict[str, Any]] = []

    def _call_one(item: Dict[str, Any]) -> Dict[str, Any]:
        if client is None:
            return {
                "view_idx": item["view_idx"],
                "azimuth": item["azimuth"],
                "elevation": item["elevation"],
                "dist": item["dist"],
                "img_path": item["img_path"],
                "raw_response": "",
                "parsed": None,
                "pred_green": None,
                "pred_yellow": None,
                "green_correct": False,
                "yellow_correct": False,
                "both_correct": False,
            }

        messages = _build_single_view_messages(
            client=client,
            prompt_text=prompt_text,
            image=item["input_img"],
            view_direction=item["view_direction"],
        )
        resp = client.call(messages, enable_thinking=False, temperature=args.temperature)
        parsed = _extract_json(resp)
        sid_map = _guess_part_names_map(parsed or {})

        green_name = (sid_map.get(green_sid) or {}).get("name", "")
        yellow_name = (sid_map.get(yellow_sid) or {}).get("name", "")
        pred_green = _normalize_lr(green_name)
        pred_yellow = _normalize_lr(yellow_name)

        green_ok = (pred_green == args.expected_green)
        yellow_ok = (pred_yellow == args.expected_yellow)
        both_ok = bool(green_ok and yellow_ok)

        # 保存原始响应
        resp_path = os.path.join(responses_dir, f"view_{item['view_idx']:03d}.txt")
        with open(resp_path, "w", encoding="utf-8") as f:
            f.write(resp)

        return {
            "view_idx": item["view_idx"],
            "azimuth": item["azimuth"],
            "elevation": item["elevation"],
            "dist": item["dist"],
            "view_direction": item["view_direction"],
            "img_path": item["img_path"],
            "response_path": resp_path,
            "parsed": parsed,
            "green_sid": green_sid,
            "yellow_sid": yellow_sid,
            "green_name": green_name,
            "yellow_name": yellow_name,
            "pred_green": pred_green,
            "pred_yellow": pred_yellow,
            "expected_green": args.expected_green,
            "expected_yellow": args.expected_yellow,
            "green_correct": green_ok,
            "yellow_correct": yellow_ok,
            "both_correct": both_ok,
        }

    if args.dry_run:
        results = [_call_one(it) for it in view_items]
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
            futs = {ex.submit(_call_one, it): it for it in view_items}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(
                    f"[DONE] view={r['view_idx']:03d} az={r['azimuth']:.1f} el={r['elevation']:.1f} "
                    f"pred(green)={r.get('pred_green')} pred(yellow)={r.get('pred_yellow')} "
                    f"both_correct={r.get('both_correct')}"
                )

    results.sort(key=lambda x: x["view_idx"])

    # 写 JSON 汇总
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(hc.convert_numpy_types(results), f, ensure_ascii=False, indent=2)

    # 写 CSV 汇总（不引入 pandas，手写）
    csv_path = os.path.join(run_dir, "summary.csv")
    csv_cols = [
        "view_idx", "azimuth", "elevation", "dist", "img_path", "response_path",
        "green_sid", "yellow_sid",
        "green_name", "yellow_name",
        "pred_green", "pred_yellow",
        "expected_green", "expected_yellow",
        "green_correct", "yellow_correct", "both_correct",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for r in results:
            row = []
            for c in csv_cols:
                v = r.get(c, "")
                s = str(v).replace("\n", " ").replace("\r", " ").replace(",", " ")
                row.append(s)
            f.write(",".join(row) + "\n")

    # Top/bottom 便于快速定位
    both_ok = [r for r in results if r.get("both_correct")]
    print(f"[SUMMARY] both_correct={len(both_ok)}/{len(results)}")
    print(f"[OUTPUT] {run_dir}")

    # 显式释放 Open3D 资源（尽量减少退出时 filament 析构崩溃概率）
    try:
        if renderer_fast is not None:
            renderer_fast.cleanup()
    except Exception:
        pass
    try:
        if renderer_final is not None:
            renderer_final.cleanup()
    except Exception:
        pass
    renderer_fast = None
    renderer_final = None
    gc.collect()

    # 仍可能在解释器退出时触发 filament 析构崩溃；默认硬退出规避
    if not args.no_hard_exit:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


