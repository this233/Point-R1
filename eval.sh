export OPENAI_API_KEY=sk-mscedQQkD6otlI9OA355F508D162439aA30f440fB62d577f
export OPENAI_API_BASE=https://api.vveai.com/v1 


export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=2 

model_name=outputs/PointLLM_train_stage3_v2/PointLLM_train_stage3

results_path=$model_name/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json


python pointllm/eval/eval_objaverse.py \
    --model_name $model_name \
    --start_eval

# python  pointllm/eval/evaluator.py \
#     --results_path $results_path \
#     --model_type gpt-4 \
#     --eval_type open-free-form-classification \
#     --parallel --num_workers 15

