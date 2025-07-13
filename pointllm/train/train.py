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
    version: Optional[str] = field(default="v1")

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
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
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
    stage_2: bool = field(default=False)
    # 预训练的多模态MLP适配器路径
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None)
    # 是否分离点标记（已弃用）
    detatch_point_token: bool = field(default=False)
    # 点云骨干网络检查点路径
    point_backbone_ckpt: str = field(default=None)

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
        model = PointLLMLlamaForCausalLM._from_config(config)
    else:
        # 正常模式：从预训练模型加载
        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # 启用Flash Attention 2
            device_map="auto"
        )

    # 打印model.dtype
    logger.info(f"1 model.dtype: {model.dtype}")
    # 禁用模型缓存
    model.config.use_cache = False

    # 根据fix_llm标志决定是否固定LLM参数
    if training_args.fix_llm:
        # 固定所有参数
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        # 固定llama、lm_head、pointnet、投影层参数
        model.requires_grad_(False)
        # 设置模型的fix_llm标志
        model.get_model().fix_llm = True
        # 设置点云投影层为可训练
        model.get_model().point_proj.requires_grad_(True)
        # 设置点云骨干网络为可训练（为FSDP设置为True，使用fix_pointnet标志控制）
        model.get_model().point_backbone.requires_grad_(True)
    else:
        # LLM可训练
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    # 从预训练模型加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 根据模型版本设置分词器和对话模板
    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        # v0版本已弃用
        raise ValueError("v0 is deprecated.")
    else:
        # print("!!!!",tokenizer.pad_token,tokenizer.unk_token)
        # 设置填充标记为未知标记
        # tokenizer.pad_token = tokenizer.unk_token
        # 设置默认对话模板
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    # 根据fix_pointnet标志决定是否固定点云骨干网络
    if not training_args.fix_pointnet:
        # 点云骨干网络可训练
        logger.info("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        # 固定点云骨干网络
        logger.info("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        # 使用torch.inference_mode控制，不使用requires_grad（为FSDP第二阶段考虑）
        model.get_model().fix_pointnet = True
        # 如果不是第二阶段训练
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            # 为第一阶段固定点云网络，第二阶段FSDP需要
            model.get_model().point_backbone.requires_grad_(False)
    
    # 根据tune_mm_mlp_adapter标志决定是否训练投影层
    if training_args.tune_mm_mlp_adapter:
        # 投影层可训练
        # 如果添加了新标记，可能需要设置embed_tokens的require_grad = True
        # 这在initialize_tokenizer_point_backbone_config中完成
        logger.info("Point projection layer is trainable.")
    else:
        # 固定投影层
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    # 根据训练阶段加载不同的配置
    if not training_args.stage_2:
        # 第一阶段：假设需要从检查点加载llm、point_backbone和投影层
        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        # 加载点云骨干网络检查点
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
        # 初始化分词器和点云骨干网络配置
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
    else:
        # 第二阶段
        # 初始化分词器和点云骨干网络配置（不包含嵌入）
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

    # 获取点云骨干网络配置
    point_backbone_config = model.get_model().point_backbone_config

    # 设置数据参数
    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    # # 启用梯度检查点
    # if training_args.gradient_checkpointing:
    #     logger.info("Enabling gradient checkpointing...")
    #     model.gradient_checkpointing_enable()
    #     logger.info("Gradient checkpointing enabled successfully.")

    # 获取不需要梯度的参数列表
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # 如果存在不需要梯度的参数
    if len(params_no_grad) > 0:
        # 如果使用FSDP
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            # 警告信息
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            # 导入FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            # 定义FSDP补丁函数
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    # 默认使用原始参数
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            # 应用补丁
            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    # 创建数据模块
    data_module = make_object_point_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)

    # 打印model.dtype
    logger.info(f"2 model.dtype: {model.dtype}")
    # 创建PointLLM训练器
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
