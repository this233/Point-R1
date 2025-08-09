PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
# data_paths="/data/liweihong/code/Point-R1/image_data/rec_jsons_processed/refcoco_train.jsonl:/data/liweihong/code/Point-R1/image_data/rec_jsons_processed/refcocop_train.jsonl:/data/liweihong/code/Point-R1/image_data/rec_jsons_processed/refcocog_train.jsonl"
# image_folders="/data/liweihong/code/Point-R1/image_data:/data/liweihong/code/Point-R1/image_data:/data/liweihong/code/Point-R1/image_data"
# model_path="/data/liweihong/code/Point-R1/models/Qwen2.5-VL-3B-Instruct"

model_name_or_path=/data/liweihong/code/Point-R1/Point-R1/outputs/PointLLM_train_stage1_v2/PointLLM_train_stage1
data_path=/data/liweihong/code/Point-R1/data/objaverse_data
anno_path=/data/liweihong/code/Point-R1/data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)


is_reward_customized_from_vlm_module=True
echo "data_paths: $data_path"
# echo "image_folders: $image_folders"
echo "anno_path: $anno_path"

export EXP_NAME="Qwen2.5-VL-3B-Instruct-rec-lora" # TODO: change this to your own experiment name
TASK_TYPE="rec"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs sbert_similarity simcse_similarity  format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true

echo "Training completed for ${EXP_NAME}"
