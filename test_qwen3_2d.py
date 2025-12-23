"""
GLB 渲染图部件识别与坐标定位脚本

使用 Qwen3-VL 模型分析 GLB 渲染图，识别主要部件并输出其坐标位置。
"""

import json
import base64
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
from openai import OpenAI


# 扩展颜色列表
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

# 默认颜色列表
COLORS = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
] + additional_colors


# ============== Prompt 模板 ==============
PART_DETECTION_PROMPT = """你是一个专业的 3D 物体分析专家。请分析这张 **GLB 3D 模型渲染图**，识别3D对象中主要部件，并给出每个部件的**边界框坐标**。

## 任务说明
1. 识别图像中 3D 模型的主要部件
2. 为每个部件标注**边界框 (bounding box)**
3. 坐标使用 **归一化格式**：x 和 y 范围为 0-1000，其中 (0, 0) 为左上角，(1000, 1000) 为右下角

## 输出格式
请严格按照以下 JSON 格式输出：

```json
[
    {
        "label": "部件名称（中文）",
        "bbox_2d": [x1, y1, x2, y2]
    },
    ...
]
```

其中 `bbox_2d` 为 `[左上角x, 左上角y, 右下角x, 右下角y]`，坐标范围 0-1000。

请开始分析：
"""

PART_DETECTION_WITH_CONTEXT_PROMPT = """你是一个专业的 3D 物体分析专家。请分析这张 **GLB 3D 模型渲染图**，识别图像中可见的所有主要部件，并给出每个部件的**边界框坐标**。

## 背景信息
- **物体名称**: {object_name}
- **物体描述**: {object_description}

## 任务说明
1. 识别图像中 3D 模型的所有可见部件
2. 为每个部件标注**边界框 (bounding box)**
3. 坐标使用 **归一化格式**：x 和 y 范围为 0-1000，其中 (0, 0) 为左上角，(1000, 1000) 为右下角

## 输出格式
请严格按照以下 JSON 格式输出：

```json
[
    {{
        "label": "部件名称（中文）",
        "bbox_2d": [x1, y1, x2, y2]
    }},
    ...
]
```

其中 `bbox_2d` 为 `[左上角x, 左上角y, 右下角x, 右下角y]`，坐标范围 0-1000。

## 注意事项
- 只标注在图像中**清晰可见**的部件
- 边界框应**紧密包围**该部件
- 部件命名应准确、具体
- 如果某个部件被遮挡或不可见，请不要标注

请开始分析：
"""


# ============== DashScope API 客户端 ==============
class DashScopeClient:
    """阿里云 DashScope API 客户端（Qwen3-VL 等模型），支持流式响应"""
    
    def __init__(self, api_key: str = None, model: str = "qwen3-vl-plus", 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 enable_thinking: bool = False, thinking_budget: int = 4096):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def call(self, messages: List[Dict], max_tokens: int = 4096, 
             stream: bool = True, **kwargs) -> str:
        """调用 DashScope API，支持流式响应和思考模式"""
        enable_thinking = kwargs.get('enable_thinking', self.enable_thinking)
        temperature = kwargs.get('temperature', 0.7)
        
        # 构建请求参数
        kwargs_api = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 如果启用思考模式
        if enable_thinking:
            kwargs_api["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": self.thinking_budget
            }
        
        if not stream:
            # 非流式调用
            completion = self.client.chat.completions.create(**kwargs_api)
            return completion.choices[0].message.content
        
        # 流式调用
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
                print(delta.content, end="", flush=True)  # 实时输出
        
        print()  # 换行
        
        # 如果需要，可以记录思考过程
        if enable_thinking and reasoning_content:
            print(f"\n[Thinking] {reasoning_content}\n")
        
        return answer_content


# ============== 图像处理函数 ==============
def encode_image_to_base64(image_path: str) -> str:
    """将本地图像编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_url(image_path: str) -> str:
    """获取图像的 URL（本地文件转为 base64 data URL）"""
    if image_path.startswith(("http://", "https://")):
        return image_path
    
    # 本地文件，转为 base64
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(ext, "image/png")
    base64_data = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{base64_data}"


# ============== JSON 解析函数 ==============
def parse_json_response(text: str) -> List[Dict]:
    """解析模型返回的 JSON 响应"""
    try:
        # 清理 markdown 标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        data = json.loads(text)
        
        if isinstance(data, dict) and "parts" in data:
            return data["parts"]
        elif isinstance(data, list):
            return data
        else:
            return [data]
    
    except Exception as e:
        print(f"[警告] JSON 解析失败: {e}")
        print(f"原始响应: {text[:500]}...")
        return []


def decode_bboxes_from_response(data: List[Dict]) -> Tuple[List[List[int]], List[str]]:
    """从解析后的数据中提取边界框和标签"""
    bboxes = []
    labels = []
    
    for item in data:
        if "bbox_2d" in item:
            bbox = item["bbox_2d"]
            bboxes.append([int(b) for b in bbox])
            label = item.get("label", f"part_{len(bboxes)}")
            labels.append(label)
    
    return bboxes, labels


# ============== 可视化函数 ==============
def plot_points_on_image(image_path: str, data: List[Dict], output_path: str = None, show: bool = True):
    """在图像上绘制检测到的部件点"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size=16)
        except:
            font = ImageFont.load_default()
    
    for i, item in enumerate(data):
        if "point_2d" not in item:
            continue
        
        color = COLORS[i % len(COLORS)]
        x_norm, y_norm = item["point_2d"]
        
        # 将归一化坐标转换为绝对坐标
        abs_x = int(x_norm / 1000 * width)
        abs_y = int(y_norm / 1000 * height)
        
        # 绘制点
        radius = 6
        draw.ellipse(
            [(abs_x - radius, abs_y - radius), (abs_x + radius, abs_y + radius)],
            fill=color, outline='white', width=2
        )
        
        # 绘制标签
        label = item.get("label", f"part_{i+1}")
        # 添加背景框使文字更清晰
        bbox = draw.textbbox((abs_x + radius + 4, abs_y - 8), label, font=font)
        draw.rectangle(bbox, fill='white')
        draw.text((abs_x + radius + 4, abs_y - 8), label, fill=color, font=font)
    
    # 保存结果
    if output_path:
        img.save(output_path)
        print(f"[保存] 可视化结果已保存到: {output_path}")
    
    # 显示图像
    if show:
        img.show()
    
    return img


def plot_bboxes_on_image(image_path: str, data: List[Dict], output_path: str = None, show: bool = True):
    """在图像上绘制检测到的边界框"""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=25)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", size=25)
        except:
            font = ImageFont.load_default()
    
    for i, item in enumerate(data):
        if "bbox_2d" not in item:
            continue
        
        color = COLORS[i % len(COLORS)]
        bbox = item["bbox_2d"]
        
        # 将归一化坐标转换为绝对坐标 [x1, y1, x2, y2]
        abs_x1 = int(bbox[0] / 1000 * width)
        abs_y1 = int(bbox[1] / 1000 * height)
        abs_x2 = int(bbox[2] / 1000 * width)
        abs_y2 = int(bbox[3] / 1000 * height)
        
        # 确保坐标顺序正确
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        # 绘制矩形框
        draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline=color, width=3)
        
        # 绘制标签
        label = item.get("label", f"part_{i+1}")
        draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)
    
    if output_path:
        img.save(output_path)
        print(f"[保存] 可视化结果已保存到: {output_path}")
    
    if show:
        img.show()
    
    return img


# ============== 主要功能函数 ==============
def detect_parts_from_glb_render(
    image_path: str,
    object_name: str = None,
    object_description: str = None,
    model: str = "qwen3-vl-plus",
    output_path: str = None,
    show: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    从 GLB 渲染图中检测部件并返回坐标
    
    Args:
        image_path: GLB 渲染图的路径（本地文件或 URL）
        object_name: 物体名称（可选，提供上下文）
        object_description: 物体描述（可选，提供上下文）
        model: 使用的模型名称
        output_path: 可视化结果保存路径
        show: 是否显示可视化结果
        verbose: 是否打印详细信息
    
    Returns:
        检测到的部件列表，每个部件包含 label, point_2d, description
    """
    # 初始化客户端
    client = DashScopeClient(model=model)
    
    # 构建 prompt
    if object_name and object_description:
        prompt = PART_DETECTION_WITH_CONTEXT_PROMPT.format(
            object_name=object_name,
            object_description=object_description
        )
    else:
        prompt = PART_DETECTION_PROMPT
    
    # 获取图像 URL
    image_url = get_image_url(image_path)
    
    # 构建消息
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是一个专业的 3D 物体分析专家，擅长识别 3D 模型的部件结构。"}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    if verbose:
        print(f"[开始] 分析图像: {image_path}")
        print(f"[模型] {model}")
        print("-" * 50)
    
    # 调用 API
    response = client.call(messages, max_tokens=4096, temperature=0.7)
    
    if verbose:
        print("-" * 50)
    
    # 解析响应
    parts_data = parse_json_response(response)
    
    if verbose:
        print(f"\n[结果] 检测到 {len(parts_data)} 个部件:")
        for i, part in enumerate(parts_data):
            label = part.get("label", "未知")
            bbox = part.get("bbox_2d", [0, 0, 0, 0])
            print(f"  {i+1}. {label}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    
    # 可视化
    if output_path or show:
        if not output_path:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_detected.png")
        plot_bboxes_on_image(image_path, parts_data, output_path=output_path, show=show)
    
    return parts_data


def detect_parts_batch(
    image_paths: List[str],
    output_dir: str = None,
    **kwargs
) -> Dict[str, List[Dict]]:
    """批量处理多张 GLB 渲染图"""
    results = {}
    
    for i, image_path in enumerate(image_paths):
        print(f"\n{'='*60}")
        print(f"[进度] 处理第 {i+1}/{len(image_paths)} 张图像")
        
        if output_dir:
            output_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.png")
        else:
            output_path = None
        
        try:
            parts = detect_parts_from_glb_render(
                image_path, 
                output_path=output_path,
                **kwargs
            )
            results[image_path] = parts
        except Exception as e:
            print(f"[错误] 处理失败: {e}")
            results[image_path] = []
    
    return results


# ============== 命令行接口 ==============
def main():
    parser = argparse.ArgumentParser(description="GLB 渲染图部件检测工具")
    parser.add_argument("image", nargs="?", help="GLB 渲染图路径")
    parser.add_argument("--name", "-n", help="物体名称（提供上下文）")
    parser.add_argument("--desc", "-d", help="物体描述（提供上下文）")
    parser.add_argument("--model", "-m", default="qwen3-vl-plus", help="模型名称")
    parser.add_argument("--output", "-o", help="输出图像路径")
    parser.add_argument("--no-show", action="store_true", help="不显示可视化结果")
    parser.add_argument("--batch", "-b", nargs="+", help="批量处理多张图像")
    parser.add_argument("--output-dir", help="批量处理时的输出目录")
    parser.add_argument("--save-json", help="保存检测结果到 JSON 文件")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        results = detect_parts_batch(
            args.batch,
            output_dir=args.output_dir,
            model=args.model,
            show=False,
        )
        
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n[保存] 结果已保存到: {args.save_json}")
    
    elif args.image:
        # 单图处理模式
        parts = detect_parts_from_glb_render(
            args.image,
            object_name=args.name,
            object_description=args.desc,
            model=args.model,
            output_path=args.output,
            show=not args.no_show,
        )
        
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(parts, f, ensure_ascii=False, indent=2)
            print(f"\n[保存] 结果已保存到: {args.save_json}")
    
    else:
        # 默认示例：使用项目中的示例图像
        example_image = "outputs/captions/images/global_view_0.png"
        if os.path.exists(example_image):
            print(f"[示例] 使用默认图像: {example_image}")
            parts = detect_parts_from_glb_render(
                example_image,
                model=args.model,
                show=not args.no_show,
            )
        else:
            parser.print_help()
            print("\n[提示] 请提供 GLB 渲染图路径，或使用 --batch 进行批量处理")


if __name__ == "__main__":
    main()
