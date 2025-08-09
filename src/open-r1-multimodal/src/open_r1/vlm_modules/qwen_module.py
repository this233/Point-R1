from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from open_r1.model import Point_R1ForCausalLM, Point_R1Config
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine


class Qwen2VLModule(VLMBaseModule):
    sbert_model = SentenceTransformer('/data/liweihong/code/Point-R1/checkpoints/Qwen2.5-VL-3B-Instruct-Point/all-mpnet-base-v2')
    simcse_tokenizer = AutoTokenizer.from_pretrained("/data/liweihong/code/Point-R1/checkpoints/Qwen2.5-VL-3B-Instruct-Point/sup-simcse-roberta-large")
    simcse_model = AutoModel.from_pretrained("/data/liweihong/code/Point-R1/checkpoints/Qwen2.5-VL-3B-Instruct-Point/sup-simcse-roberta-large")

    def __init__(self):
        super().__init__()

    
    @staticmethod
    def sbert_similarity_reward(completions, solution, **kwargs):
        """Calculate SBERT similarity reward between model output and ground truth."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            # Extract answer from content if it has think/answer tags
            import re
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            model_output = content_match.group(1).strip() if content_match else content.strip()
            
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            try:
                embeddings = Qwen2VLModule.sbert_model.encode([ground_truth, model_output])
                sbert_similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()
                rewards.append(float(sbert_similarity))
            except Exception as e:
                print(f"Error in SBERT evaluation: {e}")
                rewards.append(0.0)
        
        return rewards


    @staticmethod
    def simcse_similarity_reward(completions, solution, **kwargs):
        """Calculate SimCSE similarity reward between model output and ground truth."""
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            # Extract answer from content if it has think/answer tags
            import re
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            model_output = content_match.group(1).strip() if content_match else content.strip()
            
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            try:
                inputs = Qwen2VLModule.simcse_tokenizer([ground_truth, model_output], padding=True, truncation=True, return_tensors="pt")

                # Get the embeddings
                with torch.no_grad():
                    embeddings = Qwen2VLModule.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

                # Calculate cosine similarity
                simcse_similarity = 1 - cosine(embeddings[0], embeddings[1]) # * cosine actually calculates cosine distance, which is 1 - cosine similarity
                rewards.append(float(simcse_similarity))
            except Exception as e:
                print(f"Error in SimCSE evaluation: {e}")
                rewards.append(0.0)
        
        return rewards

    @staticmethod
    def _retry_with_backoff(func, initial_delay: float = 1.0, exponential_base: float = 2.0, jitter: bool = True, max_retries: int = 8, max_delay: float = 30.0):
        def wrapper(*args, **kwargs):
            import time, random
            delay = initial_delay
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts > max_retries:
                        raise e
                    sleep_time = min(delay, max_delay) * (1 + (random.random() if jitter else 0))
                    time.sleep(sleep_time)
                    delay *= exponential_base
        return wrapper

    @staticmethod
    def _get_open_free_form_cls_prompt():
        return (
            """Analyze two sentences and determine if they're referring to the same general object or concept, focusing on the type of object, not attributes such as color, size, or shape. Respond with 'T' if they refer to the same thing and 'F' if not. Also, provide a brief rationale (no more than 20 words) for your judgment.
Example:
Input: 1. Spiral staircase that goes from a ground floor. 2. This is a 3D model of wooden stairs in light brown
Output: T#Both refer to a staircase.

Now, analyze the following:
Input: 1. {ground_truth} 2. {model_output}
Output: """
        )

    @staticmethod
    def llm_classification_reward(completions, solution, **kwargs):
        """LLM-based open classification reward.
        Uses an LLM to judge whether model output matches the ground-truth object description.
        Returns 1.0 for 'T' and 0.0 for 'F' (invalid responses treated as 0.0).
        """
        import os, re, json
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Endpoint and credentials (reference: ModelArts MaaS chat/completions)
        base_url = os.getenv("OPENAI_API_BASE", "https://api.modelarts-maas.com/v1")
        url = f"{base_url.rstrip('/')}/chat/completions"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set; returning zero rewards.")
            return [0.0 for _ in completions]

        model_name = os.getenv("GPT_TYPE", "deepseek-r1-250528")
        prompt_tpl = Qwen2VLModule._get_open_free_form_cls_prompt()

        contents = [completion[0]["content"] for completion in completions]
        rewards = [0.0] * len(contents)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        @Qwen2VLModule._retry_with_backoff
        def chat_complete(messages):
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "temperature": 0.3,
                "top_p": 0.95,
                "max_tokens": 1024,
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), verify=False, timeout=60)
            resp.raise_for_status()
            return resp.json()

        def process_one(index, content, sol):
            try:
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                model_output = content_match.group(1).strip() if content_match else content.strip()

                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                user_prompt = prompt_tpl.format(ground_truth=ground_truth, model_output=model_output)
                messages = [{"role": "user", "content": user_prompt}]

                resp = chat_complete(messages)
                try:
                    gpt_text = resp['choices'][0]['message']['content'].strip()
                except Exception:
                    gpt_text = ""

                label = gpt_text[0].upper() if len(gpt_text) > 0 else 'F'
                reward = 1.0 if label == 'T' else 0.0
                return index, reward
            except Exception as e:
                print(f"Error in LLM classification reward (idx={index}): {e}")
                return index, 0.0

        max_workers = int(os.getenv("LLM_PARALLEL_WORKERS", "32"))
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, (content, sol) in enumerate(zip(contents, solution)):
                futures.append(executor.submit(process_one, idx, content, sol))
            for fut in as_completed(futures):
                idx, reward = fut.result()
                rewards[idx] = reward

        return rewards
    
        
    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        model_cls = Point_R1ForCausalLM
        # if "Qwen2-VL" in model_id:
        #     model_cls = Qwen2VLForConditionalGeneration
        # elif "Qwen2.5-VL" in model_id:
        #     model_cls = Qwen2_5_VLForConditionalGeneration
        # else:
        #     raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "sbert_similarity":
            return Qwen2VLModule.sbert_similarity_reward
        elif func == "simcse_similarity":
            return Qwen2VLModule.simcse_similarity_reward
        elif func == "llm_classification":
            return Qwen2VLModule.llm_classification_reward
        else:
            raise ValueError(f"Unsupported reward function: {func}")
