#!/bin/bash
#
# PartNeXt QA 数据集 GRPO (RL) 训练脚本
#
# 在 SFT 模型基础上使用 GRPO 进行强化学习训练。
#
# 使用方法:
#   1. 首先完成 SFT 训练
#   2. 修改下方的模型和数据路径
#   3. 运行: bash run_grpo_partnext.sh
#
# 奖励函数:
#   - accuracy: 答案准确性奖励
#   - format: 格式正确性奖励 (<think>...</think><answer>...</answer>)
#

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"

# ============== 配置区域 ==============
# SFT 后的模型路径
model_path="${REPO_HOME}/Point-R1/outputs/PointLLM_partnext_sft/PointLLM_train_partnext_sft_stage2"

# PartNeXt QA 数据（JSONL 格式）
data_paths="${REPO_HOME}/data/partnext_data/anno_data/partnext_qa_train.jsonl"
# 对于 GRPO，需要将 JSON 转为 JSONL 格式

# 点云目录
pointcloud_dir="${REPO_HOME}/data/partnext_data/pointclouds"

# 实验名称
EXP_NAME="PointLLM-PartNeXt-GRPO"

# 任务类型和奖励方法
TASK_TYPE="partnext"
REWARD_METHOD="default"

# GPU 数量
NUM_GPUS=8
# ======================================

cd ${REPO_HOME}/src/open-r1-multimodal

# 启用调试模式
export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

echo "======================================"
echo "PartNeXt QA GRPO Training"
echo "Model: $model_path"
echo "Data: $data_paths"
echo "Experiment: $EXP_NAME"
echo "======================================"

torchrun --nproc_per_node="$NUM_GPUS" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12350" \
    src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --data_path $pointcloud_dir \
    --anno_path $data_paths \
    --use_color True \
    --pointnum 8192 \
    --task_type $TASK_TYPE \
    --reward_method $REWARD_METHOD \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 4 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name partnext_qa \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json

echo "======================================"
echo "GRPO Training completed for ${EXP_NAME}"
echo "======================================"


