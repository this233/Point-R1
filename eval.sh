export OPENAI_API_KEY=sk-mscedQQkD6otlI9OA355F508D162439aA30f440fB62d577f
export OPENAI_API_BASE=https://api.vveai.com/v1 


export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=2 

model_name=/data/liweihong/code/Point-R1/Point-R1/outputs/PointLLM_train_stage1_v2
adapter_name=/data/liweihong/code/Point-R1/checkpoints/rl/Qwen2.5-VL-3B-Instruct-rec-lora/checkpoint-400

results_path=$adapter_name/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json


python /data/liweihong/code/Point-R1/src/open-r1-multimodal/src/open_r1/eval/eval_modelnet_cls.py \
    --model_name $model_name \
    --adapter_name $adapter_name \
    --start_eval

# python  pointllm/eval/evaluator.py \
#     --results_path $results_path \
#     --model_type gpt-4 \
#     --eval_type open-free-form-classification \
#     --parallel --num_workers 15

