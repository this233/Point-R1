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

# * 使用示例:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # 如果不在缓存中，或访问次数达到最大值，则加载并放入缓存
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    """LRU缓存类，实现最近最少使用缓存机制"""
    def __init__(self, capacity, max_access_count):
        """
        初始化LRU缓存
        :param capacity: 缓存容量
        :param max_access_count: 最大访问次数
        """
        self.cache = OrderedDict()  # 使用有序字典维护缓存项的插入顺序
        self.access_count = defaultdict(int)  # 记录每个键的访问次数
        self.capacity = capacity  # 缓存容量
        self.max_access_count = max_access_count  # 最大访问次数

    def get(self, key):
        """
        获取缓存中的值
        :param key: 键
        :return: 对应的值，如果不存在则返回None
        """
        if key not in self.cache:  # 如果键不在缓存中
            return None
        value = self.cache.pop(key)  # 取出值并删除原位置
        self.cache[key] = value  # 将键重新插入到最后（最新位置）
        self.access_count[key] += 1  # 增加访问计数
        return value

    def put(self, key, value):
        """
        向缓存中添加或更新键值对
        :param key: 键
        :param value: 值
        """
        if key in self.cache:  # 如果键已存在，更新值并移到最新位置
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # 如果缓存已满
            oldest_key = next(iter(self.cache))  # 获取最旧的键
            self.cache.popitem(last=False)  # 删除最旧的项
            del self.access_count[oldest_key]  # 删除对应的访问计数
        self.cache[key] = value  # 添加新的键值对
        self.access_count[key] = 1  # 设置初始访问计数为1

    def get_access_count(self, key):
        """
        获取指定键的访问次数
        :param key: 键
        :return: 访问次数
        """
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        """
        重置指定键的访问次数
        :param key: 键
        """
        self.access_count[key] = 0


# def preprocess_v1(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """
#     预处理对话数据的主要函数
#     :param sources: 源对话数据
#     :param tokenizer: 分词器
#     :return: 包含input_ids和labels的字典
#     """
#     # [[
#     # {'from': 'human', 'value': '<point_start>...<point_end>\nWhat kind of object is illustrated by this collection of points?'},
#     # {'from': 'gpt', 'value': 'A 3D model of a low poly yellow tree with a brown base.'}]]
#     # conv = conversation_lib.default_conversation.copy()  # 复制默认对话模板
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # 定义角色映射

#     # 应用提示模板
#     conversations = []  # 存储处理后的对话
#     for i, source in enumerate(sources):  # 遍历每个对话源
#         if roles[source[0]["from"]] != conv.roles[0]:  # 如果第一条消息不是来自人类
#             # 跳过第一条消息
#             source = source[1:]

#         conv.messages = []  # 清空对话消息
#         for j, sentence in enumerate(source):  # 遍历对话中的每句话
#             role = roles[sentence["from"]]  # 获取角色
#             assert role == conv.roles[j % 2], f"{i}"  # 确保角色交替出现
#             conv.append_message(role, sentence["value"])  # 添加消息到对话
#         conversations.append(conv.get_prompt())  # 获取格式化的对话提示  conv.get_prompt()="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <point_start><point_patch>x513<point_end>\nWhat kind of object is illustrated by this collection of points? ASSISTANT: A 3D model of a low poly yellow tree with a brown base.</s>"

#     # 对对话进行分词
#     print("!!!!", conversations)  # 调试输出
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",  # 返回PyTorch张量
#         padding="longest",  # 填充到最长序列
#         max_length=tokenizer.model_max_length,  # 最大长度限制
#         truncation=True,  # 启用截断
#     ).input_ids
#     targets = input_ids.clone()  # 复制input_ids作为目标标签
#     # [[    32,   6236,   1948,    264,  22208,   1196,    323,    458,  20443,
#     #      11229,  17847,     13,    576,  17847,   6696,  10950,     11,  11682,
#     #         11,    323,  47787,  11253,    311,    279,   1196,    594,   4755,
#     #         13,  13872,     25,    220, 151666, 151665, 151665, 151665, 151665,
#     #     151665, 151665, 151665, 151665, 151665, 151665, 151665, 151665, 151665,
#     #     ...,
#     #     151665, 151665, 151665, 151665, 151665, 151667,    198,   3838,   3093,
#     #         315,   1633,    374,  35662,    553,    419,   4426,    315,   3501,
#     #          30,  35560,   3846,   2821,     25,    362,    220,     18,     35,
#     #      1614,    315,    264,   3347,   9861,  13753,   4916,    448,    264,
#     #     13876,   2331,   3918,     82,     29]])

#     # assert conv.sep_style == conversation_lib.SeparatorStyle.TWO  # 确保使用正确的分隔符样式

#     # 掩码目标标签
#     sep = conv.sep + conv.roles[1] + ": "  # 构造分隔符
#     for conversation, target in zip(conversations, targets):  # 遍历对话和目标
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 计算非填充token的总长度

#         rounds = conversation.split(conv.sep2)  # 按轮次分割对话
#         cur_len = 1  # 当前长度
#         target[:cur_len] = IGNORE_INDEX  # 设置开始部分为忽略索引
#         for i, rou in enumerate(rounds):  # 遍历每一轮对话
#             if rou == "":  # 如果轮次为空
#                 break

#             parts = rou.split(sep)  # 分割指令和回答部分
#             if len(parts) != 2:  # 如果分割结果不是2部分（可以处理填充token）
#                 break
#             parts[0] += sep  # 给指令部分添加分隔符
#             round_len = len(tokenizer(rou).input_ids)  # 计算本轮的token长度
#             instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # 计算指令部分长度

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX  # 将指令部分设为忽略索引

#             cur_len += round_len  # 更新当前长度
#         target[cur_len:] = IGNORE_INDEX  # 将剩余部分设为忽略索引（填充token需要）

#         if cur_len < tokenizer.model_max_length:  # 如果当前长度小于最大长度
#             if cur_len != total_len:  # 如果长度不匹配（对话中的未知token会导致这种情况）
#                 target[:] = IGNORE_INDEX  # 将整个目标设为忽略索引
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,  # 输入token ID
#         labels=targets,  # 标签
#     )


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