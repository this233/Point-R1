"""
PartNeXt QA 任务的奖励函数

支持的 QA 类型:
- object_identity: 物体识别
- part_identity: 部件识别  
- part_count: 部件计数
- other: 其他问题
"""

import re
import json
from typing import List, Dict, Any, Optional
from Levenshtein import ratio


def extract_answer_content(text: str) -> str:
    """提取 <answer>...</answer> 标签中的内容"""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def extract_think_content(text: str) -> str:
    """提取 <think>...</think> 标签中的内容"""
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def clean_text(text: str) -> str:
    """清理文本，用于比较"""
    # 提取 answer 内容
    text = extract_answer_content(text)
    # 转小写并去除多余空白
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def identity_reward(content: str, solution: str, **kwargs) -> float:
    """
    物体/部件识别任务的奖励函数。
    
    评估模型是否正确识别了物体或部件的类别/名称。
    使用模糊匹配来处理表述差异。
    
    Args:
        content: 模型预测
        solution: 标准答案
        
    Returns:
        奖励分数 [0, 1]
    """
    pred = clean_text(content)
    gt = clean_text(solution)
    
    if not pred or not gt:
        return 0.0
    
    # 完全匹配
    if pred == gt:
        return 1.0
    
    # 检查是否包含关键类别词
    # 提取第一个词作为主要类别（如 "Table. A minimalist table..." -> "table"）
    pred_words = pred.split()
    gt_words = gt.split()
    
    pred_category = pred_words[0].rstrip('.:,') if pred_words else ""
    gt_category = gt_words[0].rstrip('.:,') if gt_words else ""
    
    if pred_category == gt_category:
        return 0.9
    
    # 模糊匹配
    similarity = ratio(pred, gt)
    
    # 检查是否是子串关系
    if pred_category in gt or gt_category in pred:
        similarity = max(similarity, 0.8)
    
    return similarity


def count_reward(content: str, solution: str, **kwargs) -> float:
    """
    部件计数任务的奖励函数。
    
    评估模型是否正确计数了部件数量。
    
    Args:
        content: 模型预测
        solution: 标准答案
        
    Returns:
        奖励分数 [0, 1]
    """
    pred = clean_text(content)
    gt = clean_text(solution)
    
    # 提取数字
    pred_nums = re.findall(r'\d+', pred)
    gt_nums = re.findall(r'\d+', gt)
    
    if not gt_nums:
        # 如果标准答案没有数字，使用模糊匹配
        return ratio(pred, gt)
    
    gt_num = int(gt_nums[0])
    
    if not pred_nums:
        # 尝试识别英文数字
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12
        }
        for word, num in word_to_num.items():
            if word in pred.lower():
                if num == gt_num:
                    return 1.0
                else:
                    # 根据差距给予部分分数
                    diff = abs(num - gt_num)
                    return max(0.0, 1.0 - diff * 0.2)
        return 0.0
    
    pred_num = int(pred_nums[0])
    
    if pred_num == gt_num:
        return 1.0
    
    # 根据差距给予部分分数
    diff = abs(pred_num - gt_num)
    return max(0.0, 1.0 - diff * 0.2)


def other_qa_reward(content: str, solution: str, **kwargs) -> float:
    """
    其他问题类型的奖励函数。
    
    使用模糊匹配和语义相似度。
    
    Args:
        content: 模型预测
        solution: 标准答案
        
    Returns:
        奖励分数 [0, 1]
    """
    pred = clean_text(content)
    gt = clean_text(solution)
    
    if not pred or not gt:
        return 0.0
    
    if pred == gt:
        return 1.0
    
    # 模糊匹配
    return ratio(pred, gt)


def think_quality_reward(content: str, **kwargs) -> float:
    """
    思考过程质量奖励。
    
    评估 <think> 标签中的推理过程质量。
    
    Args:
        content: 模型完整输出
        
    Returns:
        奖励分数 [0, 1]
    """
    think_content = extract_think_content(content)
    
    if not think_content:
        return 0.0
    
    # 基础分数：有思考过程
    score = 0.5
    
    # 长度奖励：适当长度的思考过程
    word_count = len(think_content.split())
    if 20 <= word_count <= 150:
        score += 0.3
    elif 10 <= word_count < 20:
        score += 0.2
    elif word_count > 150:
        score += 0.1
    
    # 结构奖励：包含多个句子
    sentence_count = len(re.findall(r'[.!?]', think_content))
    if sentence_count >= 3:
        score += 0.2
    elif sentence_count >= 2:
        score += 0.1
    
    return min(1.0, score)


def partnext_accuracy_reward(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs
) -> List[float]:
    """
    PartNeXt QA 准确性奖励函数。
    
    根据 QA 类型选择不同的评估方法。
    
    Args:
        completions: 模型生成的完成列表
        solution: 标准答案列表
        **kwargs: 额外参数，包括 qa_type
        
    Returns:
        奖励分数列表
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # 获取 qa_type（如果提供）
    qa_types = kwargs.get("qa_type", ["other"] * len(contents))
    if isinstance(qa_types, str):
        qa_types = [qa_types] * len(contents)
    
    for content, sol, qa_type in zip(contents, solution, qa_types):
        if qa_type in ("object_identity", "part_identity", "partnext_object_identity", "partnext_part_identity"):
            reward = identity_reward(content, sol)
        elif qa_type in ("part_count", "partnext_part_count"):
            reward = count_reward(content, sol)
        else:
            reward = other_qa_reward(content, sol)
        
        rewards.append(reward)
    
    return rewards


def partnext_format_reward(
    completions: List[List[Dict[str, str]]],
    **kwargs
) -> List[float]:
    """
    PartNeXt QA 格式奖励函数。
    
    检查输出是否符合 <think>...</think><answer>...</answer> 格式。
    
    Args:
        completions: 模型生成的完成列表
        **kwargs: 额外参数
        
    Returns:
        奖励分数列表
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content in contents:
        if re.fullmatch(pattern, content, re.DOTALL):
            rewards.append(1.0)
        elif re.search(r"<answer>.*?</answer>", content, re.DOTALL):
            # 有 answer 但格式不完整
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    
    return rewards


def partnext_think_reward(
    completions: List[List[Dict[str, str]]],
    **kwargs
) -> List[float]:
    """
    PartNeXt QA 思考过程奖励函数。
    
    Args:
        completions: 模型生成的完成列表
        **kwargs: 额外参数
        
    Returns:
        奖励分数列表
    """
    contents = [completion[0]["content"] for completion in completions]
    return [think_quality_reward(content) for content in contents]


def partnext_combined_reward(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs
) -> List[float]:
    """
    PartNeXt QA 综合奖励函数。
    
    组合准确性、格式和思考过程奖励。
    
    Args:
        completions: 模型生成的完成列表
        solution: 标准答案列表
        **kwargs: 额外参数
        
    Returns:
        奖励分数列表
    """
    accuracy_rewards = partnext_accuracy_reward(completions, solution, **kwargs)
    format_rewards = partnext_format_reward(completions, **kwargs)
    think_rewards = partnext_think_reward(completions, **kwargs)
    
    # 权重：准确性 0.6，格式 0.2，思考 0.2
    combined = []
    for acc, fmt, think in zip(accuracy_rewards, format_rewards, think_rewards):
        combined.append(0.6 * acc + 0.2 * fmt + 0.2 * think)
    
    return combined


# 注册奖励函数
PARTNEXT_REWARD_FUNCS = {
    "partnext_accuracy": partnext_accuracy_reward,
    "partnext_format": partnext_format_reward,
    "partnext_think": partnext_think_reward,
    "partnext_combined": partnext_combined_reward,
}


