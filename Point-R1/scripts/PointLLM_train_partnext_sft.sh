#!/bin/bash
#
# PartNeXt QA 数据集 SFT 训练脚本
#
# 使用 PartNeXt QA 数据对 PointLLM 进行监督微调 (SFT)。
#
# 使用方法:
#   1. 首先使用 convert_partnext_qa_to_pointllm.py 转换数据
#   2. 修改下方的数据路径
#   3. 运行: bash PointLLM_train_partnext_sft.sh
#
# 训练阶段:
#   - Stage 1: 对齐阶段，只训练 point_proj，冻结 LLM
#   - Stage 2: 指令微调阶段，训练 point_proj + LLM (LoRA)
#

master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
filename=$(basename "$0" | cut -f 1 -d '.')

# 设置库路径
export LDFLAGS="-Wl,--no-as-needed -ldl -laio"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# ============== 配置区域 ==============
# 基础模型路径（预训练的 PointLLM）
model_name_or_path=../checkpoints/Qwen2.5-VL-3B-Instruct-Point

# PartNeXt QA 数据路径（转换后的数据）
data_path=../data/partnext_data/pointclouds
anno_path=../data/partnext_data/anno_data/partnext_qa_train.json

# 输出目录
output_dir=outputs/PointLLM_partnext_sft/$filename

# Point backbone 检查点（如果有）
point_backbone_ckpt=../checkpoints/Qwen2.5-VL-3B-Instruct-Point/point_bert_v1.2.pt

# 训练阶段: 1 = 对齐, 2 = 指令微调
STAGE=2

# GPU 数量
NUM_GPUS=4
# ======================================

export WANDB_PROJECT="Point-R1-PartNeXt"
export PYTHONHASHSEED=0

echo "======================================"
echo "PartNeXt QA SFT Training"
echo "Stage: $STAGE"
echo "Model: $model_name_or_path"
echo "Data: $data_path"
echo "Anno: $anno_path"
echo "Output: $output_dir"
echo "======================================"

if [ "$STAGE" == "1" ]; then
    # Stage 1: 对齐阶段
    # 只训练 point_proj，冻结 LLM
    echo "Running Stage 1: Alignment (point_proj only)"
    
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$master_port \
        pointllm/train/train_mem.py \
        --model_name_or_path $model_name_or_path \
        --data_path $data_path \
        --anno_path $anno_path \
        --output_dir ${output_dir}_stage1 \
        --model_max_length 2048 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 3e-4 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --bf16 True \
        --stage 1 \
        --gradient_checkpointing True \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --run_name ${filename}_stage1 \
        --point_backbone_ckpt $point_backbone_ckpt \
        --use_color True \
        --dataloader_num_workers 8 \
        --report_to wandb \
        --max_grad_norm 1.0 \
        --seed 42 \
        --data_seed 42 \
        --llm_train_type fix \
        --train_norm False \
        --train_point_proj True \
        --train_point_backbone False \
        --conversation_types "partnext_object_identity" "partnext_part_identity" "partnext_other" "partnext_part_count"

elif [ "$STAGE" == "2" ]; then
    # Stage 2: 指令微调阶段
    # 训练 point_proj + LLM (LoRA)
    echo "Running Stage 2: Instruction Tuning (point_proj + LLM LoRA)"
    
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=$master_port \
        pointllm/train/train_mem.py \
        --model_name_or_path $model_name_or_path \
        --data_path $data_path \
        --anno_path $anno_path \
        --output_dir ${output_dir}_stage2 \
        --model_max_length 2048 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --learning_rate 1e-4 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --bf16 True \
        --stage 2 \
        --gradient_checkpointing True \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --run_name ${filename}_stage2 \
        --use_color True \
        --dataloader_num_workers 8 \
        --report_to wandb \
        --max_grad_norm 1.0 \
        --seed 42 \
        --data_seed 42 \
        --llm_train_type lora \
        --train_norm True \
        --train_point_proj True \
        --train_point_backbone False \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
        --conversation_types "partnext_object_identity" "partnext_part_identity" "partnext_other" "partnext_part_count"

else
    echo "Invalid STAGE: $STAGE. Use 1 or 2."
    exit 1
fi

echo "======================================"
echo "Training completed for ${filename}"
echo "Output saved to: $output_dir"
echo "======================================"


