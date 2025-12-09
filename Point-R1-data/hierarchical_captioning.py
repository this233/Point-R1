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

import numpy as np
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
    confidence: float  # 置信度 (0-1)
    point_count: int
    visible_ratio: float  # 在最佳视角下的可见比例
    children: List['ClusterCaption'] = field(default_factory=list)


@dataclass
class ViewQualityScore:
    """单个视角的质量评分"""
    view_idx: int
    clarity_score: float  # 清晰度 (0-10)
    completeness_score: float  # 完整性 (0-10)
    occlusion_score: float  # 遮挡程度 (0-10, 越高越好=遮挡越少)
    distinguishability_score: float  # 可区分性 (0-10)
    overall_score: float  # 总分
    reasoning: str


@dataclass
class ClusterAnnotation:
    """单个视角下的簇标注"""
    cluster_id: int
    som_id: int
    name: str
    description: str
    confidence: float
    color: str = ""
    matched_features: List[str] = field(default_factory=list)  # Step 1 中对应的视觉特征


@dataclass
class ViewAnnotationResult:
    """单个视角的标注结果"""
    view_idx: int
    quality_score: ViewQualityScore
    annotations: List[ClusterAnnotation]
    unmatched_features: List[Dict[str, str]] = field(default_factory=list)  # 无法匹配的 Step 1 特征



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
        
        # 构建请求参数
        kwargs_api = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # 使用流式以支持 thinking
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


# ===================== Prompt 模板 =====================

GLOBAL_CAPTION_PROMPT = """你是一个专业的 3D 物体分析专家。请分析以下多视角渲染图像，为这个 3D 物体提供：

1. **全局名称**: 简短的物体名称（2-5个字），如"龙形生物"、"机械手臂"、"古代建筑"
2. **全局描述**: 对物体的整体描述（50-100字），包括：
   - 物体的整体类别和形态
   - 主要组成部分
   - 显著的视觉特征
   - 可能的用途或背景

请以 JSON 格式输出：
```json
{
    "global_name": "物体名称",
    "global_caption": "详细描述...",
    "main_parts": ["部件1", "部件2", ...],
    "confidence": 0.95
}
```

### 置信度评分标准 (Confidence Score)
- **0.90 - 1.00 (非常确定)**: 特征极其清晰，典型且无歧义，你有 100% 的把握判断正确。
- **0.70 - 0.89 (确定)**: 特征清晰可见，符合一般认知，主要结构明确，但可能存在微小的不确定性。
- **0.50 - 0.69 (不太确定)**: 特征部分模糊、遮挡或有多种解释可能，只能基于现有信息做出最合理的推测。
- **< 0.50 (猜测)**: 几乎无法辨认，严重模糊或遮挡，主要靠猜测。

注意：
- 如果某些视角图像质量较差或不清晰，请在分析时降低其权重
- 描述应该客观准确，避免主观猜测
- 如果无法确定物体类型，请如实说明
- **专注于 3D 对象本身**：描述物体的形状、结构、组成部分、材质等固有属性
- **忽略视角相关信息**：不要在描述中提及"从某视角看"、"正面/背面"等视角相关内容
- **严格避免幻觉**：仅描述图像中清晰可见、确信存在的内容。对于模糊不清的细节，请直接忽略或如实描述为"不清晰"，**绝对不要编造**细节。"""


VISUAL_ANALYSIS_PROMPT = """你是一个专业的 3D 物体分析专家。请分析这张 3D 物体的渲染图像（Clean 视图），该图像展示了物体的一个特定视角。

## 背景信息 (仅供参考)
**注意：以下信息来自上一层级的分析，可能存在不准确的情况。请务必以当前图像中实际看到的视觉信息为准。如果图像内容与背景描述有冲突，请果断忽略背景描述，直接描述你所看到的。**
- **物体全局名称**: {global_name}
- **物体全局描述**: {global_caption}
- **父节点名称**: {parent_name}
- **父节点描述**: {parent_caption}
- **当前层级**: Level {current_level}
- **视角信息**: {view_info}

## 任务
请详细描述在该视角下，该 **3D 物体** 清晰可见的**主要子部件**和**视觉特征**。
**核心要求**：
- **描述 3D 物体本身**：请描述物体的形状、结构、材质、纹理等固有属性。
- **避免 2D 图像描述**：严禁使用"图像左侧"、"可以看到"、"画面中"等描述，而是使用"物体左侧"、"具有..."、"表面覆盖..."等。

请关注：
1. **几何结构**: 能够区分出来的独立形状或突起。
2. **外观细节**: 颜色、纹理、材质的变化。
3. **空间关系**: 各部分之间的相对位置（如"上部"、"底座"、"外侧"等）。

请以简洁的列表形式输出你的观察结果，不要包含任何 JSON 格式，只需自然语言描述。"""


SOM_MATCHING_PROMPT = """你是一个专业的 3D 物体分析专家。当前任务是**将视觉特征准确映射到 3D 空间区域**（SoM 标注）。

## 1. 核心参考信息 (语义与内容来源)
**请务必综合利用以下信息，这是你理解物体和获取视觉细节的关键：**

### A. 父节点约束 (语义上下文)
当前分析的部件是 **{parent_name}** 的子结构。
- **父节点描述**: {parent_caption}
- **物体整体**: {global_name}
- **指导原则**: 你的分析**必须**在父节点的语义范围内。例如，如果父节点是“头部”，那么子区域应当是“眼睛”、“嘴巴”等头部组件，而非“脚部”。请参考父节点描述来理解当前部件的功能和上下文。

### B. 视觉特征分析 (内容核心)
这是基于 Clean 视图（真实纹理渲染）的详细分析结果。**这是你获取形状、纹理、材质、颜色等外观信息的唯一真实来源**：
>>>
{visual_analysis}
<<<
**重要**：最终的标注内容（Name 和 Description）**必须**源自上述视觉分析。不要编造视觉分析中未提及的细节。

## 2. 空间定位参考 (SoM 视图)
你将看到一张**SoM 视图** (颜色编码的点云图)。
- **功能**: **仅用于界定区域范围和 ID**。
- **警告**: 图中的颜色是随机分配的 ID 颜色，**绝不代表**物体真实颜色。图中也**没有**清晰的纹理细节。
- **ID 映射表**:
{color_mapping}

## 3. 标注任务
请执行 **“视觉-空间对齐”**：将 **[视觉特征分析]** 中描述的具体部件和细节，准确填入 **[SoM 视图]** 对应的颜色区域中。

### 操作步骤
1. **理解上下文**: 阅读父节点描述，明确当前部件在整体中的角色。
2. **提取特征**: 从 [视觉特征分析] 中提取关键部件（如“金属外壳”、“红色按钮”）。
3. **空间定位**: 观察 SoM 视图，根据**几何形状**和**空间位置**（如“顶部圆形突起”、“左侧长条状”），找到这些部件对应的颜色区域。
4. **生成标注**:
   - **名称**: 结合父节点语义和视觉分析，给该区域起一个准确的名称。
   - **描述**: **必须**基于 [视觉特征分析] 中的描述撰写。描述该区域的真实材质、颜色和结构。
   - **严禁**: 不要描述 SoM 图的伪色（如“这个区域是红色的”指 SoM 颜色），除非物体真实颜色也是红色。

### 核心原则
1. **区域含义多样性**: 一个 SoM 区域可能是一个完整部件、部件的一部分，或多个部件的集合。请根据实际覆盖范围如实描述。
2. **包容性匹配**: 只要 [视觉特征分析] 中的某个特征在空间上落入某颜色区域，就应包含在该区域的描述中。
3. **未匹配处理**: 只有当某个视觉特征在 SoM 图中完全对应**灰色/背景/无编号区域**时，才列入 `unmatched_features`。请仔细检查，不要因为颜色区域包含了额外部分就错误地将其归为未匹配。

### 输出格式
请以 JSON 格式输出：
```json
{{
    "quality_assessment": {{
        "clarity_score": 8,
        "completeness_score": 7,
        "occlusion_score": 6,
        "distinguishability_score": 8,
        "overall_score": 7.25,
        "reasoning": "简要评估图像质量..."
    }},
    "annotations": [
        {{
            "som_id": 1,
            "color": "红色",
            "name": "准确名称",
            "description": "详细描述 (基于视觉分析)",
            "confidence": 0.85,
            "matched_features": ["对应视觉分析中的特征A", "对应特征B"]
        }},
        ...
    ],
    "unmatched_features": [
        {{"feature": "视觉分析中提到的背景", "reason": "对应SoM图中的灰色背景区域"}},
        {{"feature": "底部的黑色底座", "reason": "在SoM图中该位置为无编号的灰色区域，未被分割为独立部件"}}
    ]
}}
```

### 评分标准 (请严格按照以下标准打分)

#### 1. Clarity Score (清晰度)
- **0-3分**: 图像严重模糊、像素化严重或噪点过多，无法辨认细节结构。
- **4-6分**: 物体轮廓可见，但表面纹理模糊，或存在轻微的渲染伪影。
- **7-8分**: 图像清晰，物体主要结构和纹理细节清晰可见，边缘锐利。
- **9-10分**: 极其清晰，高分辨率感，微小的纹理和材质细节都能完美呈现。

#### 2. Completeness Score (完整性)
- **0-3分**: 仅可见物体的一小部分（<30%），关键结构缺失。
- **4-6分**: 物体主体部分可见（30%-70%），但部分关键区域被截断在图像边缘之外。
- **7-8分**: 物体绝大部分可见（>90%），仅边缘细微部分可能被切掉。
- **9-10分**: 物体完全包含在图像视野内，构图完美完整。

#### 3. Occlusion Score (遮挡程度)
*注意：分数越高代表遮挡越少（越好）。*
- **0-3分**: 大部分区域（>70%）被其他物体或自身部件严重遮挡，无法看清主体。
- **4-6分**: 存在明显遮挡（30%-50%），影响了对部分结构的观察。
- **7-8分**: 仅有轻微遮挡（<15%），关键特征未受影响。
- **9-10分**: 完全无遮挡，目标区域一览无余。

#### 4. Distinguishability Score (可区分性)
- **0-3分**: 目标区域与背景混杂，或特征过于普通，难以提取有效特征。
- **4-6分**: 能看出是独立部分，但缺乏显著的几何或纹理特征。
- **7-8分**: 具有明确的形状或纹理特征，易于描述和识别。
- **9-10分**: 特征极具辨识度（如独特的标志、复杂的机械结构等），一眼即可锁定。

#### 5. Confidence Score (置信度 - 针对每个标注)
- **0.90 - 1.00 (非常确定)**: 特征极其清晰，典型且无歧义，你有 100% 的把握判断正确。
- **0.70 - 0.89 (确定)**: 特征清晰可见，符合一般认知，主要结构明确，但可能存在微小的不确定性。
- **0.50 - 0.69 (不太确定)**: 特征部分模糊、遮挡或有多种解释可能，只能基于现有信息做出最合理的推测。
- **< 0.50 (猜测)**: 几乎无法辨认，严重模糊或遮挡，主要靠猜测。
"""


MERGE_ANNOTATIONS_PROMPT = """你是一个专业的 3D 物体分析专家。请整合来自多个视角的标注结果。

## 背景信息 (仅供参考)
注意：以下名称仅供参考，如果多视角标注结果中出现了更准确的描述，请优先采信标注结果。
- **物体全局名称**: {global_name}
- **父节点名称**: {parent_name}
- **子部件数量**: {num_children}

## 多视角标注结果
{multi_view_annotations}

## 任务
请基于各视角的图像质量评分和标注内容，为每个子部件生成最终的综合标注。

### 整合原则
1. **优先采信高质量视角**: 图像质量分数高的视角权重更大
2. **信息互补**: 不同视角可能看到不同细节，应综合考虑
3. **冲突处理**: 如果不同视角的描述有矛盾，优先相信质量更高的视角
4. **置信度计算**: 最终置信度应综合考虑各视角的置信度和图像质量
5. **去伪存真**：如果在某些低质量视角中出现了疑似幻觉的描述（与其他高质量视角严重不符），请果断舍弃。

### 输出格式
```json
{{
    "merged_annotations": [
        {{
            "som_id": 1,
            "name": "最终确定的名称",
            "caption": "综合多视角信息的详细描述（30-80字）",
            "confidence": 0.85,
            "best_view_idx": 2,
            "reasoning": "选择该结论的原因"
        }},
        ...
    ],
    "annotation_notes": "整体标注过程中的注意事项或不确定性说明"
}}
```

### 置信度评分标准 (Confidence Score)
- **0.90 - 1.00 (非常确定)**: 特征极其清晰，典型且无歧义，你有 100% 的把握判断正确。
- **0.70 - 0.89 (确定)**: 特征清晰可见，符合一般认知，主要结构明确，但可能存在微小的不确定性。
- **0.50 - 0.69 (不太确定)**: 特征部分模糊、遮挡或有多种解释可能，只能基于现有信息做出最合理的推测。
- **< 0.50 (猜测)**: 几乎无法辨认，严重模糊或遮挡，主要靠猜测。

### 重要：描述内容要求
- **专注于 3D 对象本身**：最终 caption 应描述部件的形状、结构、功能、材质等固有属性
- **忽略视角相关信息**：不要在 caption 中提及任何视角、角度、可见性等信息
- **描述应具有普遍性**：生成的描述应该是对该部件的客观、全面的描述，而非特定视角下的观察结果
- **严格避免幻觉**：最终输出的描述必须是所有视角中**证据确凿**的交集。对于不确定的细节，宁可不写，也不要猜测。"""


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
    betas: List[float] = [0.0, 0.3, 0.5, 0.7]
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Any]:
    """
    加载数据并执行聚类
    
    返回:
        points: 归一化后的点云坐标
        clustering_results: 各层级的聚类标签
        model: 归一化后的 GLB 模型
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
    
    return points, clustering_results, model


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


def generate_sibling_views(
    points: np.ndarray,
    clustering_results: Dict[int, np.ndarray],
    model: Any,
    parent_level_idx: int,
    parent_id: int,
    image_size: int = 800,
    num_views: int = 4
) -> Tuple[List[Dict], Dict[int, int], int, str]:
    """
    为兄弟簇组生成多视角渲染
    
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
    print(f"    [DEBUG generate_sibling_views] 子簇点数详情: {child_point_counts}")
    print(f"    [DEBUG generate_sibling_views] SoM ID 映射: {som_id_map}")
    
    # 收集组内所有点
    group_indices = []
    group_point_child_ids = []
    for cid in child_ids:
        indices = np.where(child_labels == cid)[0]
        group_indices.extend(indices)
        group_point_child_ids.extend([cid] * len(indices))
    
    group_points = points[group_indices]
    group_point_child_ids = np.array(group_point_child_ids)
    group_center = np.mean(group_points, axis=0)
    
    # 生成颜色映射 - 使用预定义的高区分度颜色 (20色)
    # 确保颜色之间有足够的视觉区分度
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
    
    color_map = {}
    for i, cid in enumerate(child_ids):
        color_map[cid] = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
    
    print(f"    [DEBUG] 颜色映射 (使用高区分度颜色, 共{len(DISTINCT_COLORS)}种):")
    for cid in child_ids:
        som_id = som_id_map[cid]
        color = color_map[cid]
        count = np.sum(child_labels == cid)
        print(f"      - 簇 ID {cid} (SoM ID {som_id}): {count} 点, 颜色 RGB={color}")
    
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
    
    view_points = sample_view_points(radius=1.0, partition=3)
    candidates = []
    
    for vp in view_points:
        direction = vp / np.linalg.norm(vp)
        dist = optimize_distance_for_cluster(
            renderer, group_points, direction, group_center, 
            intrinsic, 256, target_occupancy=0.6, min_dist_threshold=0.5
        )
        
        eye = group_center + direction * dist
        extrinsic = create_camera_extrinsic_from_viewpoint(eye, center=group_center)
        
        _, depth_map = renderer.render_view(eye, center=group_center, return_depth=True)
        pixel_coords, depths, valid_mask_fov = project_points_to_image_with_depth(
            group_points, intrinsic, extrinsic, image_size=256
        )
        
        is_visible_depth = np.zeros(len(group_points), dtype=bool)
        fov_indices = np.where(valid_mask_fov)[0]
        
        if len(fov_indices) > 0:
            visible_mask_sub = check_visible_points_with_depth(
                group_points[fov_indices],
                pixel_coords[fov_indices],
                depths[fov_indices],
                depth_map,
                use_relative_threshold=True,
                relative_threshold_ratio=0.02
            )
            is_visible_depth[fov_indices] = visible_mask_sub
        
        valid_mask = is_visible_depth
        
        # 计算评分
        child_vis_list = []
        for cid in child_ids:
            c_mask = (group_point_child_ids == cid)
            c_total = np.sum(c_mask)
            c_visible = np.sum(valid_mask & c_mask)
            c_vis_ratio = c_visible / c_total if c_total > 0 else 0
            child_vis_list.append(c_vis_ratio)
        
        min_child_vis = min(child_vis_list) if child_vis_list else 0
        mean_child_vis = np.mean(child_vis_list) if child_vis_list else 0
        overall_vis = np.sum(valid_mask) / len(group_points)
        
        if overall_vis < 0.1:
            continue
        
        score = (min_child_vis * 10.0) + (mean_child_vis * 5.0) + (overall_vis * 3.0)
        
        candidates.append({
            'score': score,
            'direction': direction,
            'dist': dist,
            'eye': eye,
            'stats': {
                'min_child': min_child_vis,
                'mean_child': mean_child_vis,
                'overall': overall_vis,
                'child_vis_detail': child_vis_list
            }
        })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 筛选视角
    MIN_WORST_CHILD_VIS = 0.15
    MIN_MEAN_CHILD_VIS = 0.35
    
    valid_candidates = [c for c in candidates 
                       if c['stats']['min_child'] >= MIN_WORST_CHILD_VIS 
                       and c['stats']['mean_child'] >= MIN_MEAN_CHILD_VIS]
    
    if not valid_candidates:
        valid_candidates = [c for c in candidates 
                          if c['stats']['min_child'] >= 0.08 
                          and c['stats']['mean_child'] >= 0.20]
    
    if not valid_candidates and candidates:
        valid_candidates = candidates[:num_views]
    
    # 选择多样化视角
    DISTINCTNESS_THRESHOLD = 0.7
    final_views = []
    for cand in valid_candidates:
        if len(final_views) >= num_views:
            break
        is_distinct = all(np.dot(cand['direction'], s['direction']) <= DISTINCTNESS_THRESHOLD 
                         for s in final_views)
        if is_distinct:
            final_views.append(cand)
    
    renderer.cleanup()
    
    if not final_views:
        return [], som_id_map, child_level_idx, "无法找到有效视角"
    
    # ========== 关键修复：分离 GLB 渲染和点云渲染，避免渲染器冲突 ==========
    # 参考 visualize_pointcloud_gradio.py 中的实现
    
    # Step 1: 先渲染所有 Clean 视图 (GLB)
    renderer_final = Open3DRenderer(width=image_size, height=image_size)
    renderer_final.setup()
    renderer_final.upload_model(model)
    
    clean_images = []
    for view in final_views:
        clean_img, _ = renderer_final.render_view(view['eye'], center=group_center)
        clean_images.append(clean_img)
    
    # Step 2: 清理 GLB 渲染器（重要：必须在点云渲染之前清理）
    renderer_final.cleanup()
    renderer_final = None
    
    # Step 3: 渲染所有 SoM 视图 (点云) - 此时 GLB 渲染器已释放
    output_views = []
    for i, view in enumerate(final_views):
        clean_img = clean_images[i]
        
        # 渲染 SoM 图像（点云聚类可视化）
        som_img = render_pointcloud_som_image(
            points, child_labels, child_ids, color_map,
            view['eye'], group_center, image_size=image_size,
            point_size=3.0, dim_factor=0.25, distance=view['dist']
        )
        
        # 计算视角描述
        vec = view['eye'] - group_center
        dist = np.linalg.norm(vec)
        elevation = np.degrees(np.arcsin(vec[1] / dist)) if dist > 1e-6 else 0
        azimuth = np.degrees(np.arctan2(vec[2], vec[0]))
        if azimuth < 0:
            azimuth += 360
        
        view_info = {
            'view_idx': i,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': dist,
            'eye': view['eye'].tolist(),
            'center': group_center.tolist(),
            'stats': view['stats']
        }
        
        output_views.append({
            'clean_image': clean_img,
            'som_image': som_img,
            'view_info': view_info,
            'child_ids': child_ids,
            'som_id_map': som_id_map,
            'color_map': color_map
        })
    
    return output_views, som_id_map, child_level_idx, f"生成了 {len(output_views)} 个视角"


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
    # 构建消息
    content = [{"type": "text", "text": GLOBAL_CAPTION_PROMPT}]
    
    for i, img in enumerate(images):
        img_b64 = client.encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
        content.append({"type": "text", "text": f"[视角 {i+1}]"})
    
    messages = [{"role": "user", "content": content}]
    
    print(f"\n{'='*20} MLLM INPUT (Global Caption) {'='*20}", flush=True)
    print(GLOBAL_CAPTION_PROMPT, flush=True)
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
        "main_parts": [],
        "confidence": 0.5
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


def call_mllm_for_sibling_annotation(
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
    color_map: Dict[int, List[int]]
) -> Tuple[ViewAnnotationResult, Dict]:
    """调用 MLLM 获取单个视角的标注（两步法：视觉分析 -> 匹配标注）"""
    
    view_idx = view_info.get('view_idx', '?')
    
    # 1. 准备基础信息
    azimuth = view_info['azimuth']
    elevation = view_info['elevation']
    view_direction = get_view_direction_description(azimuth, elevation)
    view_info_str = f"- 视角: {view_direction}"
    
    # ================= Step 1: 视觉分析 (基于 Clean View) =================
    
    analysis_prompt = VISUAL_ANALYSIS_PROMPT.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        current_level=current_level,
        view_info=view_info_str
    )
    
    content_step1 = [
        {"type": "text", "text": analysis_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(clean_image)}"}}
    ]
    
    messages_step1 = [{"role": "user", "content": content_step1}]
    
    print(f"\n{'='*20} MLLM INPUT (Step 1: Visual Analysis - View {view_idx}) {'='*20}", flush=True)
    print(analysis_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Step 1: 不需要 Thinking
    analysis_response = client.call(messages_step1, enable_thinking=False)
    
    print(f"\n{'='*20} MLLM OUTPUT (Step 1) {'='*20}", flush=True)
    print(analysis_response, flush=True)
    print(f"{'='*60}\n", flush=True)

    # ================= Step 2: 匹配与标注 (基于 Combined View) =================

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
    
    matching_prompt = SOM_MATCHING_PROMPT.format(
        global_name=global_name,
        global_caption=global_caption,
        parent_name=parent_name,
        parent_caption=parent_caption,
        visual_analysis=analysis_response,
        color_mapping=color_mapping_str
    )
    
    # Step 2: 只发送 SoM 视图
    # combined_image = np.concatenate([clean_image, som_image], axis=1) # 不再拼接
    
    content_step2 = [
        {"type": "text", "text": matching_prompt},
        {"type": "text", "text": "[图像: SoM 视图 (仅几何形状与区域编码，颜色为区域ID)]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{client.encode_image(som_image)}"}}
    ]
    
    messages_step2 = [{"role": "user", "content": content_step2}]
    
    print(f"\n{'='*20} MLLM INPUT (Step 2: SoM Matching - View {view_idx}) {'='*20}", flush=True)
    print(matching_prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Step 2: 需要 Thinking
    response = client.call(messages_step2, enable_thinking=True)
    
    print(f"\n{'='*20} MLLM OUTPUT (Step 2) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry = {
        "type": "sibling_view_annotation",
        "view_idx": view_idx,
        "step1_prompt": analysis_prompt,
        "step1_response": analysis_response,
        "step2_prompt": matching_prompt,
        "input_images": ["clean_image (Step 1)", "som_image (Step 2)"],
        "output_response": response
    }
    
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
            
            # 将解析结果加入日志，方便后续查看
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


def merge_multi_view_annotations(
    client: MLLMClient,
    view_results: List[ViewAnnotationResult],
    global_name: str,
    parent_name: str,
    child_ids: List[int],
    som_id_map: Dict[int, int]
) -> Tuple[List[Dict], Dict]:
    """合并多视角标注结果"""
    log_entry = {
        "type": "merge_annotations",
        "input_prompt": "",
        "output_response": "",
        "parsed_result": []
    }
    
    if not view_results:
        return [], log_entry
    
    # 格式化多视角标注
    multi_view_str = ""
    for vr in view_results:
        multi_view_str += f"\n### 视角 {vr.view_idx + 1}\n"
        multi_view_str += f"- 质量评分: {vr.quality_score.overall_score:.1f}/10\n"
        multi_view_str += f"- 评分详情: 清晰度={vr.quality_score.clarity_score}, "
        multi_view_str += f"完整性={vr.quality_score.completeness_score}, "
        multi_view_str += f"遮挡={vr.quality_score.occlusion_score}, "
        multi_view_str += f"可区分性={vr.quality_score.distinguishability_score}\n"
        multi_view_str += f"- 评估说明: {vr.quality_score.reasoning}\n"
        multi_view_str += "- 标注结果:\n"
        for ann in vr.annotations:
            multi_view_str += f"  - ID {ann.som_id} [{ann.color}]: {ann.name} - {ann.description} (置信度: {ann.confidence:.2f})\n"
    
    prompt = MERGE_ANNOTATIONS_PROMPT.format(
        global_name=global_name,
        parent_name=parent_name,
        num_children=len(child_ids),
        multi_view_annotations=multi_view_str
    )
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    print(f"\n{'='*20} MLLM INPUT (Merge Annotations) {'='*20}", flush=True)
    print(prompt, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Merge: 需要 Thinking
    response = client.call(messages, enable_thinking=True)
    
    print(f"\n{'='*20} MLLM OUTPUT (Merge Annotations) {'='*20}", flush=True)
    print(response, flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 记录日志
    log_entry["input_prompt"] = prompt
    log_entry["output_response"] = response
    
    # 解析结果
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            merged_list = result.get('merged_annotations', [])
            log_entry["parsed_result"] = merged_list
            return merged_list, log_entry
    except json.JSONDecodeError:
        pass
    
    # 如果解析失败，使用简单的合并策略
    merged = {}
    for vr in view_results:
        weight = vr.quality_score.overall_score / 10.0
        for ann in vr.annotations:
            if ann.som_id not in merged:
                merged[ann.som_id] = {
                    'som_id': ann.som_id,
                    'name': ann.name,
                    'caption': ann.description,
                    'confidence': ann.confidence * weight,
                    'total_weight': weight
                }
            else:
                if ann.confidence * weight > merged[ann.som_id]['confidence']:
                    merged[ann.som_id]['name'] = ann.name
                    merged[ann.som_id]['caption'] = ann.description
                merged[ann.som_id]['confidence'] += ann.confidence * weight
                merged[ann.som_id]['total_weight'] += weight
    
    # 归一化置信度
    for k in merged:
        if merged[k]['total_weight'] > 0:
            merged[k]['confidence'] /= merged[k]['total_weight']
        del merged[k]['total_weight']
    
    result_list = list(merged.values())
    log_entry["parsed_result"] = result_list
    return result_list, log_entry


# ===================== 主流程 =====================

def process_hierarchical_captioning(
    glb_path: str,
    npy_path: str,
    feature_path: str,
    output_dir: str,
    mllm_client: Optional[MLLMClient],
    k_neighbors: int = 5,
    betas: List[float] = [0.0, 0.3, 0.5, 0.7],
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
    points, clustering_results, model = load_and_prepare_data(
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
    root_cluster = ClusterCaption(
        cluster_id=0,
        som_id=0,
        level=0,
        parent_id=None,
        name=global_name,
        caption=global_caption,
        confidence=global_result.get('confidence', 0.8),
        point_count=len(points),
        visible_ratio=1.0,
        children=[]
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
            parent_level, parent_cluster_id,
            image_size=800, num_views=4
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
        
        # 保存视图图像
        # 注意：现在只保存拼接后的 Combined Image (Clean|SoM)
        # 但 generate_sibling_views 返回的是 separate 的，我们需要在这里拼接并保存
        if save_images:
            for i, view in enumerate(views):
                if view['som_image'] is not None:
                    combined = np.concatenate([view['clean_image'], view['som_image']], axis=1)
                    Image.fromarray(combined).save(
                        os.path.join(images_dir, f"L{parent_level}_C{parent_cluster_id}_combined_{i}.png")
                    )
        
        # 准备日志列表
        interaction_logs = []
        
        # 多视角标注
        if dry_run:
            print(f"    [DRY RUN] 跳过 MLLM 多视角标注调用")
            # 生成模拟的标注结果
            merged_annotations = []
            for i, cid in enumerate(valid_child_ids):
                som_id = som_id_map.get(cid, i + 1)
                merged_annotations.append({
                    'som_id': som_id,
                    'name': f'调试部件_{som_id}',
                    'caption': f'这是 dry_run 模式生成的占位描述 (簇ID={cid})',
                    'confidence': 0.8,
                    'color': '未指定'
                })
            print(f"    [DRY RUN] 生成了 {len(merged_annotations)} 个模拟标注")
        else:
            view_results = []
            for view in views:
                if view['som_image'] is None:
                    continue
                
                result, log = call_mllm_for_sibling_annotation(
                    mllm_client,
                    view['clean_image'],
                    view['som_image'],
                    global_name,
                    global_caption,
                    parent_node.name,
                    parent_node.caption,
                    child_level_idx,  # 使用实际的子层级
                    view['view_info'],
                    valid_child_ids,
                    som_id_map,
                    view['color_map']  # 传递颜色映射
                )
                view_results.append(result)
                interaction_logs.append(log)
                
                print(f"    视角 {view['view_info']['view_idx']}: 质量={result.quality_score.overall_score:.1f}")
            
            # 合并标注
            merged_annotations, log = merge_multi_view_annotations(
                mllm_client, view_results, global_name, parent_node.name,
                valid_child_ids, som_id_map
            )
            interaction_logs.append(log)
            
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
            
            child_node = ClusterCaption(
                cluster_id=cluster_id,
                som_id=som_id,
                level=child_level_idx,  # 使用实际的子层级
                parent_id=parent_cluster_id,
                name=ann.get('name', f'部件{som_id}'),
                caption=ann.get('caption', ''),
                confidence=ann.get('confidence', 0.5),
                point_count=point_count,
                visible_ratio=visible_ratio,
                children=[]
            )
            
            parent_node.children.append(child_node)
            queue.append((child_node, child_level_idx, cluster_id))  # 使用实际的子层级
            total_clusters += 1
            max_level_reached = max(max_level_reached, child_level_idx)
            
            print(f"    + {child_node.name}: {point_count} points, conf={child_node.confidence:.2f}")
    
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
    parser.add_argument('--glb_path', type=str, required=True,
                       help='GLB 文件路径')
    parser.add_argument('--npy_path', type=str, required=True,
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
    parser.add_argument('--betas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7],
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
