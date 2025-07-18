#    Copyright 2023 Runsen Xu
# 版权声明：2023年 Runsen Xu

from typing import List, Optional, Tuple, Union
# 导入类型提示相关的模块，用于函数参数和返回值的类型声明
import os
MY_DEBUG = os.getenv("MY_DEBUG", "False") == "True"

import torch
# 导入PyTorch深度学习框架
import torch.nn as nn
# 导入PyTorch神经网络模块
from torch.nn import CrossEntropyLoss
# 导入交叉熵损失函数
from .utils import *
# 导入当前包下的utils模块中的所有函数和类
from pointllm.utils import *
# 导入pointllm包下utils模块中的所有函数和类

from transformers import LlamaConfig

from contextlib import nullcontext
# 导入空上下文管理器，用于条件性地应用上下文管理器
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, Qwen2Tokenizer
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig,Qwen2_5_VLTextConfig, \
                                           Qwen2_5_VLTextModel,Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformerPretrainedModel
                                           
# 导入Transformers库中的自动配置、自动模型、Llama配置、Llama模型和Llama因果语言模型

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# 导入Transformers库中的模型输出类，用于返回模型的输出结果
import sys
import os
# 导入操作系统接口模块，用于文件路径操作

# * add logger
# 添加日志记录器
import logging
# 导入日志模块
logger = logging.getLogger(__name__)
# 创建当前模块的日志记录器实例
# 设置为INFO级别
logger.setLevel(logging.INFO)

class Point_R1TextConfig(Qwen2_5_VLTextConfig):
    # 定义PointLLM配置类，继承自Qwen2_5_VLConfig
    model_type = "point_r1_text"

class Point_R1TextModel(Qwen2_5_VLTextModel):
    # 定义PointLLM的Llama模型类，继承自LlamaModel
    config_class = Point_R1TextConfig 
    # 指定配置类为PointLLMConfig

    def init_point_proj(self):
        # * print relevant info with projection layers
        # 打印投影层相关信息
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        # 获取骨干网络输出维度
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        # 记录点云骨干网络输出维度
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        # 记录使用的投影隐藏层数量
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # 如果投影隐藏层数量大于0
            # Add projection layer with linear layers and GELU activation
            # 添加带有线性层和GELU激活函数的投影层
            projection_layers = []
            # projection_layers.append(nn.LayerNorm(backbone_output_dim, eps=1e-5))
            # 创建投影层列表
            last_dim = backbone_output_dim
            # 设置上一层的维度为骨干网络输出维度
            for i in range(self.point_bert_config.model.projection_hidden_layer):
                # 遍历每个投影隐藏层
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                # 添加线性层
                projection_layers.append(nn.GELU())
                # 添加GELU激活函数
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]
                # 更新上一层维度

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            # 添加最后的投影层
            self.point_proj = nn.Sequential(*projection_layers)
            # 创建顺序模型作为点云投影器
            logger.info(f"Each layer with {self.point_bert_config.model.projection_hidden_dim} hidden units.")
            # 记录每层的隐藏单元数量
        else:
            # 否则（没有投影隐藏层）
            # Single layer
            # 单层投影
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
            # 创建单层线性投影
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")
        # 记录点云投影器输出维度
        
        
    def __init__(self, config: Qwen2_5_VLTextConfig):
        # 初始化方法，接收LlamaConfig类型的配置参数
        super(Point_R1TextModel, self).__init__(config)
        # 调用父类的初始化方法

        self.point_backbone_type = config.point_backbone
        # 从配置中获取点云骨干网络类型
        logger.info(f"Using {self.point_backbone_type}.")
        # 记录使用的点云骨干网络类型

        if self.point_backbone_type == "PointBERT":
            # 如果点云骨干网络类型是PointBERT
            from pointllm.model import PointTransformer
            # 导入PointTransformer模型
            # address of config file, in the same dir of this file
            # 配置文件地址，在当前文件的同一目录下
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            # 获取PointBERT配置文件名，默认为v1.2版本的配置，v1.1版本使用不同的配置
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            # 构建PointBERT配置文件的完整路径
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            # 打印加载PointBERT配置文件的信息
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            self.point_bert_config = point_bert_config
            # 从YAML文件中加载PointBERT配置
            # print(point_bert_config,flush=True)
            if getattr(config, "use_color", False):
                # 如果配置中启用了颜色信息
                point_bert_config.model.point_dims = 6
                # 设置点的维度为6（包含XYZ坐标和RGB颜色）
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            # 获取是否使用最大池化的配置，默认为False
            # print(point_bert_config,flush=True)
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            # 创建PointTransformer骨干网络实例
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")
            # 记录使用的点云维度信息

            self.point_backbone_config = {
                # 创建点云骨干网络配置字典
                "point_cloud_dim": point_bert_config.model.point_dims,
                # 点云维度
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                # 骨干网络输出维度，如果使用最大池化则维度翻倍
                "project_output_dim": self.config.hidden_size,
                # 投影输出维度，等于隐藏层大小
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                # 点云token长度，包含CLS token，如果使用最大池化则为1
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                # 是否使用点云开始和结束token
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                # 投影隐藏层数量，默认为0
                "use_max_pool": use_max_pool
                # 是否使用最大池化
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                # 如果投影隐藏层数量大于0
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
                # 添加投影隐藏层维度配置（是一个列表）
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")
            # 记录最大池化使用情况和点云token数量

        self.init_point_proj()

        self.fix_pointnet = False
        # 设置是否固定PointNet的参数为False
        self.fix_llm = False
        # 设置是否固定LLM的参数为False
        self.post_init()
    
    def post_init(self):
        super(Point_R1TextModel, self).post_init()
        # # 对于nn.Linear层，避免梯度爆炸
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        #         if module.bias is not None:
        #             nn.init.zeros_(module.bias)

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        # 加载点云骨干网络检查点的方法
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)
        # 加载检查点，如果没有指定路径则使用配置中的路径

    def forward(
        # 前向传播方法
        self,
        # 自身引用
        input_ids: torch.LongTensor = None,
        # 输入token的ID，长整型张量
        attention_mask: Optional[torch.Tensor] = None,
        # 注意力掩码，可选的张量
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，用于缓存，可选的浮点张量列表
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入嵌入，可选的浮点张量
        use_cache: Optional[bool] = None,
        # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔值
        point_clouds: Optional[torch.FloatTensor] = None,
        # 点云数据，可选的浮点张量
        return_dict: Optional[bool] = None,
        # 是否返回字典格式，可选的布尔值
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 返回类型为元组或带有过去状态的基础模型输出

        # HACK: replace back original embeddings for pretraining
        # 技巧：为预训练替换回原始嵌入
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # 获取原始嵌入参数，如果不存在则返回None

        if inputs_embeds is None:
            # 如果输入嵌入为None
            inputs_embeds = self.embed_tokens(input_ids)
            # 使用输入ID生成嵌入

        point_backbone = getattr(self, 'point_backbone', None)
        # 获取点云骨干网络，如果不存在则返回None
        point_backbone_config = getattr(self, 'point_backbone_config', None)
        # 获取点云骨干网络配置，如果不存在则返回None

        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # 如果点云骨干网络存在且（输入ID序列长度不为1或正在训练）且点云数据不为None
            # * enter when training or the first generation step of inference
            # 在训练时或推理的第一个生成步骤时进入
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                # 如果固定PointNet则不计算梯度，否则使用空上下文
                if self.fix_pointnet:
                    # 如果固定PointNet
                    self.point_backbone.eval()
                    # 设置点云骨干网络为评估模式
                if type(point_clouds) is list:
                    # 如果点云数据是列表类型
                    # * variable numbers of points
                    # 可变数量的点
                    point_features = []
                    # 创建点特征列表
                    for point_cloud in point_clouds: # * iterate over batch
                        # 遍历批次中的每个点云
                        point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                        # 通过点云骨干网络处理点云，获取特征
                        point_features.append(point_feature)
                        # 将特征添加到列表中
                else:
                    # 否则（点云数据是张量）
                    point_features = self.point_backbone(point_clouds) # 16x513x384 BxLxC
                    # 直接通过点云骨干网络处理点云

            if type(point_clouds) is list:
                # 如果点云数据是列表类型
                point_features = [self.point_proj(point_feature) for point_feature in point_features]
                # 对每个点特征进行投影
            else:
                # 否则
                point_features = self.point_proj(point_features)# 16x513x2048 BxLxC
                # 直接对点特征进行投影

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype) # 513 x 384
            # 创建虚拟点特征张量，用于保持计算图的连接
            dummy_point_features = self.point_proj(dummy_point_features) # 513x2048 LxC
            # 对虚拟点特征进行投影

            new_input_embeds = []
            # 创建新的输入嵌入列表
            cur_point_idx = 0
            # 当前点云索引
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L 16x575; input_embeds: B, L, C 16x575x2048
                # 遍历每个样本的输入ID和输入嵌入 cur_input_ids: 575; cur_input_embeds: 575x2048
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # 如果当前输入中没有点云补丁token
                    # multimodal LLM, but the current sample is not multimodal
                    # 多模态LLM，但当前样本不是多模态的
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    # 添加虚拟点特征的零贡献，保持计算图连接
                    new_input_embeds.append(cur_input_embeds)
                    # 将当前输入嵌入添加到新列表中
                    cur_point_idx += 1
                    # 递增点云索引
                    continue
                    # 继续下一个样本
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device) # 513x2048 LxC
                # # DEBUG
                # cur_point_features = torch.cat((cur_point_features[0:1], cur_point_features[1::16]), dim=0)
                # 获取当前点云特征并移动到正确的设备
                num_patches = cur_point_features.shape[0] # * number of point tokens 513(CLS + 512)
                # 获取点token的数量
                if point_backbone_config['mm_use_point_start_end']:
                    # 如果使用点云开始和结束token
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        # 如果点云开始token和结束token的数量不相等
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                        # 抛出值错误异常
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    # 找到所有点云开始token的位置
                    for point_start_token_pos in point_start_tokens:
                        # 遍历每个点云开始token的位置
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            # 如果点云结束token不在预期位置
                            raise ValueError("The point end token should follow the point start token.")
                            # 抛出值错误异常
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            # 如果有原始嵌入参数，则除了点云开始和结束token外不更新原始嵌入（走这条路）
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                            # 拼接新的输入嵌入，保持某些部分不可训练
                        else:
                            # 否则
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                            # 直接拼接新的输入嵌入
                        cur_point_idx += 1
                        # 递增点云索引
                    new_input_embeds.append(cur_new_input_embeds)
                    # 将新的输入嵌入添加到列表中
                else:
                    # 否则（不使用点云开始和结束token）
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        # 如果点云补丁token的数量与点补丁数量不匹配
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                        # 抛出值错误异常
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    # 找到所有点云补丁token的位置
                    mask_index_start = masked_indices[0]
                    # 获取掩码开始位置
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        # 如果点云补丁token不是连续的
                        raise ValueError("The point patch tokens should be consecutive.")
                        # 抛出值错误异常
                    if orig_embeds_params is not None:
                        # 如果有原始嵌入参数
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                        # 拼接新的输入嵌入，保持某些部分不可训练
                    else:
                        # 否则
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                        # 直接拼接新的输入嵌入
                    new_input_embeds.append(cur_new_input_embeds)
                    # 将新的输入嵌入添加到列表中
                    cur_point_idx += 1
                    # 递增点云索引
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            # 将新的输入嵌入堆叠成张量

        return super(Point_R1TextModel, self).forward(
            # 调用父类的前向传播方法
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            # 传递参数：输入ID设为None，注意力掩码，过去的键值对
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            # 传递输入嵌入和缓存使用标志
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            # 传递注意力和隐藏状态输出标志
            return_dict=return_dict
            # 传递返回字典标志
        )


class Point_R1Config(Qwen2_5_VLConfig):
    # 定义PointLLM配置类，继承自Qwen2_5_VLConfig
    model_type = "point_r1"


class Point_R1Model(Qwen2_5_VLModel):
    config_class = Point_R1Config
    
    def __init__(self, config):
        super(Qwen2_5_VLModel, self).__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Point_R1TextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)

class Point_R1ForCausalLM(Qwen2_5_VLForConditionalGeneration):
    config_class = Point_R1Config
    def __init__(self, config):
        # 初始化方法
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        # 调用LlamaForCausalLM的初始化方法
        self.model = Point_R1Model(config)
        # 创建PointLLM模型实例

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False) # 2048 x 151936
        # 创建语言模型头部，用于生成词汇表大小的输出

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()
        # 调用后初始化方法

    def get_model(self):
        # 获取模型的方法
        return self.model
        # 返回模型实例

    def forward(
        # 前向传播方法
        self,
        # 自身引用
        input_ids: torch.LongTensor = None,
        # 输入token的ID Bx572(L)
        attention_mask: Optional[torch.Tensor] = None,
        # 注意力掩码 Bx572(L) input_ids.ne(self.tokenizer.pad_token_id)
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入嵌入 None
        labels: Optional[torch.LongTensor] = None,
        # 标签，用于计算损失 Bx572(L)
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        # 是否使用缓存，控制是否返回过去的键值对 None
        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重 None
        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态 None
        point_clouds: Optional[torch.FloatTensor] = None,
        # 点云数据 Bx8192x6
        return_dict: Optional[bool] = None,
        # 是否返回字典格式 None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 返回类型为元组或带有过去状态的因果语言模型输出
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # False
        # 设置注意力输出标志，如果未指定则使用配置中的值
        output_hidden_states = (
            # 设置隐藏状态输出标志
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            # 如果未指定则使用配置中的值
        ) # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True
        # 设置返回字典标志，如果未指定则使用配置中的值

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 解码器输出包含（解码特征，层状态，解码隐藏状态，解码注意力）
        outputs = self.model(
            # 调用模型的前向传播
            input_ids=input_ids,
            # 传递输入ID
            attention_mask=attention_mask,
            # 传递注意力掩码
            past_key_values=past_key_values,
            # 传递过去的键值对
            inputs_embeds=inputs_embeds,
            # 传递输入嵌入
            use_cache=use_cache,
            # 传递缓存使用标志
            output_attentions=output_attentions,
            # 传递注意力输出标志
            output_hidden_states=output_hidden_states,
            # 传递隐藏状态输出标志
            return_dict=return_dict,
            # 传递返回字典标志
            point_clouds=point_clouds
            # 传递点云数据
        )

        hidden_states = outputs[0] # 16x593x2048
        # 获取隐藏状态
        logits = self.lm_head(hidden_states) # 16x593x151668
        # 通过语言模型头部计算logits

        loss = None
        # 初始化损失为None
        if labels is not None:
            # 如果标签存在
            # Shift so that tokens < n predict n
            # 移位使得tokens < n预测n
            # print(f"=== DEBUG INFO START ===", file=sys.stderr, flush=True)
            # print(f"logits.shape: {logits.shape}", file=sys.stderr, flush=True)
            # print(f"labels.shape: {labels.shape}", file=sys.stderr, flush=True)
            # print(f"logits.dtype: {logits.dtype}", file=sys.stderr, flush=True)
            # print(f"labels.dtype: {labels.dtype}", file=sys.stderr, flush=True)
            # print(f"self.config.vocab_size: {self.config.vocab_size}", file=sys.stderr, flush=True)
            # print(f"=== DEBUG INFO END ===", file=sys.stderr, flush=True)

            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            # 移位logits，形状为（批次，长度-1，词汇表大小）
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # 移位标签，形状为（批次，长度-1）
            
            # print(f"=== AFTER SHIFT DEBUG ===", file=sys.stderr, flush=True)
            # print(f"shift_logits.shape: {shift_logits.shape}", file=sys.stderr, flush=True)
            # print(f"shift_labels.shape: {shift_labels.shape}", file=sys.stderr, flush=True)
            # print(f"shift_logits.numel(): {shift_logits.numel()}", file=sys.stderr, flush=True)
            # print(f"Expected reshape size: {shift_logits.numel() // self.config.vocab_size}", file=sys.stderr, flush=True)
            # print(f"Remainder: {shift_logits.numel() % self.config.vocab_size}", file=sys.stderr, flush=True)
            # print(f"=== END AFTER SHIFT DEBUG ===", file=sys.stderr, flush=True)
            
            actual_vocab_size = self.lm_head.weight.shape[0]
            # 由于添加token，或者其他因素，config中的vocab_size可能不准确
            # Flatten the tokens
            # 展平tokens
            loss_fct = CrossEntropyLoss()
            # 创建交叉熵损失函数
            shift_logits = shift_logits.view(-1, actual_vocab_size)
            # 将logits重塑为（-1，词汇表大小）
            shift_labels = shift_labels.view(-1)
            # 将标签重塑为一维
            # Enable model/pipeline parallelism
            # 启用模型/管道并行
            shift_labels = shift_labels.to(shift_logits.device)
            # 将标签移动到与logits相同的设备
            loss = loss_fct(shift_logits, shift_labels)
            # 计算损失

        if not return_dict:
            # 如果不返回字典格式
            output = (logits,) + outputs[1:]
            # 组合输出
            return (loss,) + output if loss is not None else output
            # 返回损失和输出，如果损失存在

        return CausalLMOutputWithPast(
            # 返回因果语言模型输出
            loss=loss,
            # 损失
            logits=logits,
            # logits
            past_key_values=outputs.past_key_values,
            # 过去的键值对
            hidden_states=outputs.hidden_states,
            # 隐藏状态
            attentions=outputs.attentions,
            # 注意力权重
        )

    def prepare_inputs_for_generation(
        # 为生成准备输入的方法
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        # 输入参数：输入ID，过去的键值对，注意力掩码，输入嵌入和其他关键字参数
    ):
        if past_key_values:
            # 如果存在过去的键值对
            input_ids = input_ids[:, -1:]
            # 只保留最后一个token

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果传递了输入嵌入，我们只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            # 如果输入嵌入存在且没有过去的键值对
            model_inputs = {"inputs_embeds": inputs_embeds}
            # 设置模型输入为输入嵌入
        else:
            # 否则
            model_inputs = {"input_ids": input_ids}
            # 设置模型输入为输入ID

        model_inputs.update(
            # 更新模型输入
            {
                "past_key_values": past_key_values,
                # 过去的键值对
                "use_cache": kwargs.get("use_cache"),
                # 使用缓存标志
                "attention_mask": attention_mask,
                # 注意力掩码
                "point_clouds": kwargs.get("point_clouds", None),
                # 点云数据
            }
        )
        return model_inputs
        # 返回模型输入

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # 初始化分词器和点云骨干网络配置（不包含嵌入）的方法
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        # 在阶段2或推理或无预训练推理时调用，假设分词器已有点token
        config = self.config
        # 获取配置
        point_backbone_config = self.get_model().point_backbone_config
        # 获取点云骨干网络配置
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end
        # 设置是否使用点云开始和结束token

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        # 获取默认点云补丁token

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)
        # 向分词器添加点云补丁token

        # * assert tokenizer has the default_point_patch_token
        # 断言分词器拥有默认点云补丁token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        # 设置默认点云补丁token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]
        # 设置点云补丁token的ID

        if mm_use_point_start_end:
            # 如果使用点云开始和结束token
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            # 获取默认点云开始token
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            # 获取默认点云结束token
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            # 向分词器添加点云开始和结束token

            point_backbone_config['default_point_start_token'] = default_point_start_token
            # 设置默认点云开始token
            point_backbone_config['default_point_end_token'] = default_point_end_token
            # 设置默认点云结束token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            # 设置点云开始token的ID
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
            # 设置点云结束token的ID
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):
        # 初始化分词器和点云骨干网络配置的方法
        # 初始len(tokenizer):151665 
        # https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html
        # https://github.com/wangxso/Qwen2/blob/main/tokenization_note_zh.md
        # 初始embedding大小：[151936, 2048] 
        # 151936的原因与内存计算效率有关（151643个普通token+3个特殊token+205 个 extra token=151851）（得是128的倍数，得到151936）
        # https://github.com/QwenLM/Qwen/issues/482
        config = self.config
        # 获取配置
        point_backbone_config = self.get_model().language_model.point_backbone_config
        # 获取点云骨干网络配置
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end
        # 设置是否使用点云开始和结束token

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        # 获取默认点云补丁token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        # 设置默认点云补丁token
         # 151665
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        # 向分词器添加点云补丁token，无需更新嵌入因为会被替换
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        # 调整token嵌入大小，这会使token再次可训练
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]
        # 设置点云补丁token的ID

        if mm_use_point_start_end:
            # 如果使用点云开始和结束token
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            # 获取默认点云开始token
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            # 获取默认点云结束token
            point_backbone_config['default_point_start_token'] = default_point_start_token
            # 设置默认点云开始token
            point_backbone_config['default_point_end_token'] = default_point_end_token
            # 设置默认点云结束token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            # 向分词器添加点云开始和结束token，返回新增token数量
            self.resize_token_embeddings(len(tokenizer))
            # 调整token嵌入大小
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            # 设置点云开始token的ID
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
            # 设置点云结束token的ID

            if num_new_tokens > 0:
                # 如果有新增token
                input_embeddings = self.get_input_embeddings().weight.data
                # 获取输入嵌入权重数据
                output_embeddings = self.get_output_embeddings().weight.data
                # 获取输出嵌入权重数据

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    # 计算除新token外的输入嵌入平均值
                    dim=0, keepdim=True)
                    # 在第0维上计算平均值，保持维度
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    # 计算除新token外的输出嵌入平均值
                    dim=0, keepdim=True)
                    # 在第0维上计算平均值，保持维度

                input_embeddings[-num_new_tokens:] = input_embeddings_avg.detach()
                # 用平均值初始化新的输入嵌入
                output_embeddings[-num_new_tokens:] = output_embeddings_avg.detach()
                # 用平均值初始化新的输出嵌入

                # need to update the input embeding, but no need to update the output embedding
                # 需要更新输入嵌入，但不需要更新输出嵌入
                for p in self.get_input_embeddings().parameters():
                    # 遍历输入嵌入的参数
                    p.requires_grad = True
                    # 设置为可训练
                if fix_llm:
                    # 如果固定LLM
                    self.get_model().language_model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    # 保存原始嵌入参数，只调优新的嵌入
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        # 遍历输出嵌入（LLM头部）的参数
                        p.requires_grad = False
                        # 设置为不可训练
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                    # 打印设置信息
                else:
                    # 否则
                    self.get_model().language_model.orig_embeds_params = None
                    # 不保存原始嵌入参数
                    for p in self.get_output_embeddings().parameters():
                        # 遍历输出嵌入的参数
                        p.requires_grad = True
                        # 设置为可训练
                    print("Setting output embeddings and all input embeddings trainable.")
                    # 打印设置信息

AutoConfig.register("point_r1", Point_R1Config)
# 注册PointLLM配置类到AutoConfig
AutoModelForCausalLM.register(Point_R1Config, Point_R1ForCausalLM)
# 注册PointLLM因果语言模型到AutoModelForCausalLM
