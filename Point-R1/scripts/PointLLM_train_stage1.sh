master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

# Fix library path warnings
# export LDFLAGS="-Wl,--no-as-needed -ldl -laio -Wl,-rpath,/usr/lib/x86_64-linux-gnu -Wl,-rpath,/root/miniconda/envs/pointllm/lib"
export LDFLAGS="-Wl,--no-as-needed -ldl -laio"

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

model_name_or_path=../checkpoints/Qwen2.5-VL-3B-Instruct-Point
data_path=../data/objaverse_data
anno_path=../data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
output_dir=outputs/PointLLM_train_stage1_v1/$filename
point_backbone_ckpt=../checkpoints/Qwen2.5-VL-3B-Instruct-Point/point_bert_v1.2.pt

# PYTHONPATH=.:$PYTHONPATH \

export WANDB_PROJECT="Point-R1"


# Set random seed for reproducibility
export PYTHONHASHSEED=0

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# export MY_DEBUG=True
# python pointllm/train/train_mem.py \
# torchrun --nnodes=1 --nproc_per_node=2 --master_port=$master_port pointllm/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --model_max_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --stage 1 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --run_name $filename \
    --point_backbone_ckpt $point_backbone_ckpt \
    --use_color True \
    --dataloader_num_workers 9 \
    --report_to wandb \
    --max_grad_norm 1.0  \
    --seed 42 \
    --data_seed 42 \
    --run_name stage1-point_proj_and_embedding_and_pointbackbone \
    --llm_train_type fix \
    --train_norm False \
    --train_point_backbone False \
    --train_point2Qformer_proj True \
    --train_Qformer_lora_norm False \
    --train_Qformer2token_proj True \
    --train_query_tokens True \
    --max_steps 3000

    # --evaluation_strategy "no" \
    # --max_grad_norm 1.0 \

        