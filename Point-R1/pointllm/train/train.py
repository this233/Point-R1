# 采用自 https://github.com/lm-sys/FastChat。以下是原始版权声明：
# 采用自 tatsu-lab@stanford_alpaca。以下是原始版权声明：
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# 导入logging模块并设置全局默认级别
import os
MY_DEBUG = os.getenv("MY_DEBUG", "False") == "True"

import logging
logging.basicConfig(level=logging.INFO)

import torch

# 导入数据类相关模块
from dataclasses import dataclass, field
# 导入路径处理模块
import pathlib
# 导入类型提示模块
from typing import Optional, List

# 导入transformers库
import transformers
# 导入自定义的PointLLM训练器
from pointllm.train.pointllm_trainer import PointLLMTrainer

# LoRA/PEFT支持
from peft import get_peft_model, LoraConfig, PeftModel

# 导入对话相关模块
from pointllm import conversation as conversation_lib
# 导入模型相关模块
from pointllm.model import *
# 导入数据处理模块
from pointllm.data import make_object_point_data_module

# 导入日志构建工具
from pointllm.utils import build_logger

# 定义忽略索引常量，用于损失计算时忽略某些标记
IGNORE_INDEX = -100

# 定义默认的特殊标记
DEFAULT_PAD_TOKEN = "[PAD]"      # 填充标记
DEFAULT_EOS_TOKEN = "</s>"       # 结束标记
DEFAULT_BOS_TOKEN = "</s>"       # 开始标记
DEFAULT_UNK_TOKEN = "<unk>"      # 未知标记


@dataclass
class ModelArguments:
    """模型参数配置类"""
    # 模型名称或路径
    model_name_or_path: Optional[str] = field(default="")
    # 模型版本
    # version: Optional[str] = field(default="v1")

@dataclass
class DataArguments:
    """数据参数配置类"""
    # 训练数据路径
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    # 注释数据路径，如果为None则默认使用referit3d
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    # 是否使用颜色信息
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    # 调试模式下使用的数据数量，大于0时使用调试模式
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    # 是否分割训练集和验证集
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    # 训练集和验证集的分割比例
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    # 点云中点的数量
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    # 使用的对话类型列表
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    # 是否为多模态模型
    is_multimodal: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练参数配置类，继承自transformers.TrainingArguments"""
    # 可参考 https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    # 缓存目录
    cache_dir: Optional[str] = field(default=None)
    # 优化器类型
    optim: str = field(default="adamw_torch")
    # 模型最大序列长度
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 是否使用小模型进行调试
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."})
    # 是否固定LLM参数
    # fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    llm_train_type: str = field(default="fix", metadata={"help": "LLM train type. Can be 'fix' or 'lora'."})
    # 是否固定PointNet参数
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    # 是否移除未使用的列
    remove_unused_columns: bool = field(default=False)
    # 是否强制使用FSDP
    force_fsdp: bool = field(default=False)

    # 两阶段训练相关参数
    # 是否调整多模态MLP适配器，预训练时设为True，微调时设为False
    tune_mm_mlp_adapter: bool = field(default=True)
    # 是否为第二阶段训练，微调时设为True
    stage: int = field(default=1)
    # 预训练的多模态MLP适配器路径
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None)
    # 是否分离点标记（已弃用）
    detatch_point_token: bool = field(default=False)
    # 点云骨干网络检查点路径
    point_backbone_ckpt: str = field(default=None)
    # LoRA相关参数
    # lora_enable: bool = field(default=False, metadata={"help": "是否启用LoRA微调"})
    lora_r: int = field(default=16, metadata={"help": "LoRA秩"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha参数"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", metadata={"help": "逗号分隔的LoRA注入模块名，如q_proj,v_proj"})

    train_norm: bool = field(default=False, metadata={"help": "Whether to train the norm."})
    train_point_proj: bool = field(default=False, metadata={"help": "Whether to train the point projection layer."})
    train_point_backbone: bool = field(default=False, metadata={"help": "Whether to train the point backbone."})
    train_extra_embedding: bool = field(default=False, metadata={"help": "Whether to train the extra embedding."})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """为HuggingFace训练器安全保存模型"""
    # 获取模型状态字典
    state_dict = trainer.model.state_dict()
    # 如果应该保存模型
    if trainer.args.should_save:
        # 将状态字典转移到CPU
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        # 删除原始状态字典以释放内存
        del state_dict
        # 保存模型
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    """主要的训练函数"""
    # 创建参数解析器，解析模型参数、数据参数和训练参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    # 解析命令行参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志级别为info（默认为passive/warning）
    training_args.log_level = "info"
    # 构建日志器
    logger = build_logger(__name__, training_args.output_dir + '/train.log')

    # 根据是否为调试模式选择不同的模型加载方式
    if training_args.model_debug:
        # 调试模式：不加载检查点，从配置文件加载
        config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
            )
        # 从配置创建模型
        model = Point_R1ForCausalLM._from_config(config)
    else:
        # 正常模式：从预训练模型加载
        model = Point_R1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # 启用Flash Attention 2
            # device_map="auto" # https://blog.gitcode.com/efce4e021ffded42cd16766125e1cd20.html 当使用device_map="auto"时，会干扰accelerate的分布式训练准备过程
        )

    model.requires_grad_(False)
    # 根据llm_train_type标志决定是否固定LLM参数
    if training_args.llm_train_type == "fix":
        # 固定所有参数
        logger.info("LLM is fixed. llm_train_type is set to 'fix'")
        # 固定llama、lm_head、pointnet、投影层参数
        # 设置模型的fix_llm标志
        model.get_model().language_model.llm_train_type = "fix"
        
    elif training_args.llm_train_type == "lora":
        # LLM可训练
        model.get_model().language_model.llm_train_type = "lora"
        logger.warning("LLM is trainable. llm_train_type is set to 'lora'")

        if training_args.stage == 2:
            lora_target_modules = [x.strip() for x in training_args.lora_target_modules.split(",") if x.strip()]
            for name, param in model.named_modules():
                print(f"Parameter {name}")
            lora_target_modules_regex = ".*\.language_model\..*\.("
            first=True
            for module in lora_target_modules:
                if first:
                    lora_target_modules_regex += module
                    first=False
                else:
                    lora_target_modules_regex += "|" + module
            lora_target_modules_regex += ")"
            print(lora_target_modules_regex)
            # lora_target_modules = ['language_model\.layers.*\.'+x for x in lora_target_modules]
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=lora_target_modules_regex,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            print(model)
        elif training_args.stage == 3:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
            print(f"load peft model from {model_args.model_name_or_path}")
            print(model)
            
            # 确保预训练的LoRA参数可训练
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    logger.info(f"LoRA parameter {name} is set to trainable")
    
    if training_args.train_point_proj:
        model.get_model().language_model.point_proj.requires_grad_(True)
    if training_args.train_point_backbone:
        model.get_model().language_model.point_backbone.requires_grad_(True)
    if training_args.train_extra_embedding:
        model.get_model().language_model.extra_embedding.requires_grad_(True)

    if training_args.train_norm:
        for layer in model.get_model().language_model.layers:
            layer.input_layernorm.requires_grad_(True)
            layer.post_attention_layernorm.requires_grad_(True)
        model.get_model().language_model.norm.requires_grad_(True)

    # 从预训练模型加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 根据tune_mm_mlp_adapter标志决定是否训练投影层
    if training_args.tune_mm_mlp_adapter:
        # 投影层可训练
        # 如果添加了新标记，可能需要设置embed_tokens的require_grad = True
        # 这在initialize_tokenizer_point_backbone_config中完成
        logger.info("Point projection layer is trainable.")
    else:
        # 固定投影层
        model.get_model().language_model.point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    # 根据训练阶段加载不同的配置
    if training_args.stage == 1:   
        # 第一阶段：假设需要从检查点加载llm、point_backbone和投影层
        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        # 加载点云骨干网络检查点
        model.get_model().language_model.load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
        # 初始化分词器和点云骨干网络配置
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device)
    else:
        # 第二阶段
        # 初始化分词器和点云骨干网络配置（不包含嵌入）
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

    # 获取点云骨干网络配置
    point_backbone_config = model.get_model().language_model.point_backbone_config

    # 设置数据参数
    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    # 创建数据模块
    data_module = make_object_point_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)

    # 打印model.dtype
    logger.info(f"2 model.dtype: {model.dtype}")
    # logger.info(f"2 model.model.language_model.point_backbone dtype: {model.get_model().language_model.point_backbone.dtype}")
    # 创建PointLLM训练器

    print("#########################")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} will be updated. size: {param.size()}")
        

    print("#########################")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Number of trainable parameters: {num_trainable_params}")

    trainer = PointLLMTrainer(model=model,
                    processing_class=tokenizer,
                    args=training_args,
                    **data_module)

    # 检查是否存在检查点
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # 从检查点恢复训练
        trainer.train(resume_from_checkpoint=True)
    else:
        # 开始新的训练
        trainer.train()
    # 保存训练状态
    trainer.save_state()
    # 安全保存模型
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    # 如果直接运行此脚本，调用train函数
    train()
