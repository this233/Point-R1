export OPENAI_API_KEY=sk-0xdfGKYi0W6KOzcGC4B3958f6b6b482f8616A7E05eCa7aEb 
export OPENAI_API_BASE=https://api.gpt.ge/v1 


export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=2 

model_name=outputs/PointLLM_train_stage1_v6/PointLLM_train_stage1

results_path=$model_name/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json


python pointllm/eval/eval_objaverse.py \
    --model_name $model_name \
    --start_eval \
    --gpt_type "gpt-4"

# python  pointllm/eval/evaluator.py \
#     --results_path $results_path \
#     --model_type gpt-4 \
#     --eval_type open-free-form-classification \
#     --parallel --num_workers 15

