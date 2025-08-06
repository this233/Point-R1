master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

# Fix library path warnings
# export LDFLAGS="-Wl,--no-as-needed -ldl -laio -Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,-rpath,/root/miniconda/envs/pointllm/lib"
export LDFLAGS="-Wl,--no-as-needed -ldl -laio"

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"


model_name_or_path=outputs/PointLLM_train_stage2_v2/PointLLM_train_stage2
data_path=data/objaverse_data
anno_path=data/anno_data/PointLLM_complex_instruction_70K.json # or PointLLM_brief_description_660K.json (including val sets)
output_dir=outputs/PointLLM_train_stage3/$filename

# PYTHONPATH=.:$PYTHONPATH \

export WANDB_PROJECT="Point-R1"


# Set random seed for reproducibility
export PYTHONHASHSEED=0

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# export MY_DEBUG=True
# python pointllm/train/train_mem.py \
# torchrun --nnodes=1 --nproc_per_node=2 --master_port=$master_port pointllm/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node=7 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --model_max_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --save_steps 2400 \
    --stage 3 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --llm_train_type lora \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --run_name $filename \
    --use_color True \
    --dataloader_num_workers 9 \
    --report_to wandb \
    --max_grad_norm 1.0  \
    --seed 42 \
    --data_seed 42 \
    --run_name stage3-point_proj_llmlora_norm \
    --train_norm True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj


    # --evaluation_strategy "no" \
    # --max_grad_norm 1.0 \

        