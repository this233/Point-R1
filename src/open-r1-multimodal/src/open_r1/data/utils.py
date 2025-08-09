# 导入必要的库和模块
from collections import OrderedDict, defaultdict  # 有序字典和默认字典

import transformers  # Hugging Face transformers库
# from pointllm import conversation as conversation_lib  # 导入对话处理模块
from dataclasses import dataclass  # 数据类装饰器
from typing import Optional, Dict, Sequence  # 类型提示
import torch  # PyTorch深度学习框架

import numpy as np  # 数值计算库
import os  # 操作系统接口

# 忽略索引常量，用于标记不需要计算损失的标记
IGNORE_INDEX = -100


def preprocess_v2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    预处理对话数据的主要函数，使用Qwen2.5-VL的apply_chat_template
    :param sources: 源对话数据
    :param tokenizer: 分词器
    :return: 包含input_ids和labels的字典
    """
    # 转换数据格式为标准的messages格式
    conversations = []
    for source in sources:
        messages = []
        for sentence in source:
            # 转换角色名称
            role = "user" if sentence["from"] == "human" else "assistant"
            messages.append({
                "role": role,
                "content": sentence["value"]
            })
        conversations.append(messages)

    # 使用apply_chat_template生成格式化文本
    formatted_conversations = []
    for messages in conversations:
        try:
            # 使用tokenizer的apply_chat_template方法
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # 先不分词，只生成文本
                add_generation_prompt=False  # 不添加生成提示
            )
            formatted_conversations.append(formatted_text)
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # 如果apply_chat_template失败，回退到简单拼接
            formatted_text = ""
            for msg in messages:
                formatted_text += f"{msg['role']}: {msg['content']}\n"
            formatted_conversations.append(formatted_text)
    

    # 对格式化后的对话进行分词
    input_ids = tokenizer(
        formatted_conversations,
        return_tensors="pt",  # 返回PyTorch张量
        padding="longest",  # 填充到最长序列
        max_length=tokenizer.model_max_length,  # 最大长度限制
        truncation=True,  # 启用截断
    ).input_ids
    targets = input_ids.clone()  # 复制input_ids作为目标标签

    # 掩码目标标签 - 只训练assistant的回答部分
    for i, (messages, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 计算非填充token的总长度
        
        # 重新生成完整对话文本来计算位置
        full_text = formatted_conversations[i]
        
        # 初始化所有位置为IGNORE_INDEX
        target[:] = IGNORE_INDEX
        
        # 逐个处理每条消息，使用Qwen模板格式
        for msg in messages:
            if msg["role"] == "assistant":
                # 构建assistant消息的完整模板格式
                # Qwen格式：<|im_start|>assistant\n{content}<|im_end|>\n
                assistant_pattern = f"<|im_start|>assistant\n{msg['content']}<|im_end|>"
                
                # 在完整文本中找到这个模式
                pattern_start = full_text.find(assistant_pattern)
                if pattern_start != -1:
                    # 找到assistant内容的开始位置（跳过<|im_start|>assistant\n）
                    content_start_in_pattern = len("<|im_start|>assistant\n")
                    content_start = pattern_start + content_start_in_pattern
                    content_end = content_start + len(msg['content'])
                    
                    # 分词来获取准确的token位置
                    prefix_text = full_text[:content_start]
                    content_text = msg['content']
                    suffix_text = "<|im_end|>"  # 包含结束标记
                    
                    # 分别分词前缀、内容和后缀
                    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False).input_ids
                    content_tokens = tokenizer(content_text, add_special_tokens=False).input_ids
                    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False).input_ids
                    
                    # 计算token位置
                    content_start_token = len(prefix_tokens)
                    content_end_token = content_start_token + len(content_tokens)
                    # 包含<|im_end|>标记在训练目标中
                    end_with_suffix_token = content_end_token + len(suffix_tokens)
                    
                    # 确保不超出序列长度
                    content_start_token = min(content_start_token, total_len)
                    end_with_suffix_token = min(end_with_suffix_token, total_len)
                    
                    # 只对assistant内容和结束标记计算损失
                    if content_start_token < total_len:
                        target[content_start_token:end_with_suffix_token] = input_ids[i][content_start_token:end_with_suffix_token]

    return dict(
        input_ids=input_ids,  # 输入token ID
        labels=targets,  # 标签
    )

def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    """
    预处理多模态点云数据
    :param sources: 源数据序列
    :param point_backbone_config: 点云骨干网络配置
    :param point_indicator: 点云指示符
    :return: 处理后的源数据
    """
    point_token_len = point_backbone_config['point_token_len']  # 点云token长度
    default_point_patch_token = point_backbone_config['default_point_patch_token']  # 默认点云补丁token

    for source in sources:  # 遍历每个源
        for sentence in source:  # 遍历每句话
            replace_token = default_point_patch_token * point_token_len  # 构造替换token
            # # DEBUG
            # replace_token = default_point_patch_token * (1+64)
            if point_backbone_config['mm_use_point_start_end']:  # 如果使用开始和结束token
                replace_token = point_backbone_config['default_point_start_token'] + replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)  # 替换点云指示符

    return sources

def pc_norm(pc):
    """
    点云归一化函数
    :param pc: 点云数据，形状为NxC
    :return: 归一化后的点云，形状为NxC
    """
    xyz = pc[:, :3]  # 提取XYZ坐标
    other_feature = pc[:, 3:]  # 提取其他特征

    centroid = np.mean(xyz, axis=0)  # 计算重心
    xyz = xyz - centroid  # 平移到原点
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))  # 计算最大距离
    xyz = xyz / m  # 缩放到单位球内

    pc = np.concatenate((xyz, other_feature), axis=1)  # 合并坐标和特征
    return pc

def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    """
    加载Objaverse数据集中的点云
    :param data_path: 数据路径
    :param object_id: 对象ID
    :param pointnum: 点的数量
    :param use_color: 是否使用颜色信息
    :return: 点云数据
    """
    filename = f"{object_id}_{pointnum}.npy"  # 构造文件名
    point_cloud = np.load(os.path.join(data_path, filename))  # 加载点云数据

    # 归一化
    point_cloud = pc_norm(point_cloud)

    if not use_color:  # 如果不使用颜色
        point_cloud = point_cloud[:, :3]  # 只保留XYZ坐标

    return point_cloud

@dataclass
class DataCollatorForPointTextDataset(object):
    """混合文本和点云数据集的数据整理器"""

    tokenizer: transformers.PreTrainedTokenizer  # 分词器

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        :param instances: 实例序列
        :return: 整理后的批次数据
        """
        # 提取input_ids和labels
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels")) # len(input_ids) = len(labels) = 64
        # 填充input_ids到相同长度
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,  # 批次维度在前
            padding_value=self.tokenizer.pad_token_id)  # 使用pad_token_id填充(151643:<|endoftext|>)
        # 填充labels到相同长度
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,  # 批次维度在前
                                                 padding_value=IGNORE_INDEX)  # 使用IGNORE_INDEX填充
        # 构造批次字典
        batch = dict(
            input_ids=input_ids,  # 输入token ID
            labels=labels,  # 标签
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),  # 注意力掩码 Bx572(L)
        )

        # 如果实例中包含点云数据
        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]  # 提取点云数据
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds):  # 如果所有点云形状相同
                batch['point_clouds'] = torch.stack(point_clouds)  # 堆叠成张量
            else:
                batch['point_clouds'] = point_clouds  # 保持列表形式（点云形状不同）

        return batch

def farthest_point_sample(point, npoint):
    """
    最远点采样算法
    :param point: 点云数据，形状为[N, D]
    :param npoint: 采样点数
    :return: 采样后的点云，形状为[npoint, D]
    """
    N, D = point.shape  # 获取点数和维度
    xyz = point[:,:3]  # 提取XYZ坐标
    centroids = np.zeros((npoint,))  # 初始化采样点索引
    distance = np.ones((N,)) * 1e10  # 初始化距离数组
    farthest = np.random.randint(0, N)  # 随机选择第一个点
    for i in range(npoint):  # 采样npoint个点
        centroids[i] = farthest  # 记录当前最远点
        centroid = xyz[farthest, :]  # 获取当前最远点坐标
        dist = np.sum((xyz - centroid) ** 2, -1)  # 计算所有点到当前点的距离
        mask = dist < distance  # 找到距离更近的点
        distance[mask] = dist[mask]  # 更新最小距离
        farthest = np.argmax(distance, -1)  # 找到下一个最远点
    point = point[centroids.astype(np.int32)]  # 根据索引提取采样点
    return point

def pc_normalize(pc):
    """
    点云归一化函数
    :param pc: 点云数据，形状为Nx3
    :return: 归一化后的点云
    
    此函数将点云归一化到单位球内。
    首先计算点云的重心并将其平移到原点，
    然后缩放所有点使其适合单位球内。
    """
    centroid = np.mean(pc, axis=0)  # 计算重心
    pc = pc - centroid  # 平移到原点
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 计算最大距离
    pc = pc / m  # 缩放到单位球
    return pc