import json
import argparse
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import time
import random

# Configuration
API_ENDPOINT = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
MODEL_NAME = "gpt-4o" # Or user's preferred model

PROMPT_TEMPLATE = """
You are an expert in evaluating 3D Multi-modal Large Language Models.
The system takes a 3D point cloud as input and generates a text description. Then a classifier (GPT-4) predicts the category based on that description.

We are analyzing a failure case where the **Predicted Label** does not match the **Ground Truth Label**.

**Case Details:**
- **Ground Truth (GT)**: "{gt_label}"
- **Predicted Label (Pred)**: "{pred_label}"
- **Model Description**: "{model_output}"

**Task:**
Analyze *why* the error occurred based on the Model Description. Categorize the error into exactly ONE of the following detailed categories:

1. **Visual: Shape-Similar Confusion**
   - The description portrays an object that **geometrically resembles** the GT object (e.g., both are boxy, cylindrical, or flat), but interprets it as a different class.
   - *Example*: Describing a boxy `radio` as a `microwave` or `oven`; Describing a `stool` as a `vase` (if cylindrical).

2. **Visual: Severe Hallucination**
   - The description portrays an object that is **completely different** from the GT in both shape and semantics. The model "sees" something that isn't there at all.
   - *Example*: Describing a `plant` as a `car`; Describing a `person` as a `bookshelf`.

3. **Visual: Part Hallucination/Miss**
   - The model correctly identifies the general category or main body but **hallucinates specific parts** that don't exist (causing a class shift) or **misses key parts**.
   - *Example*: Describing a `stool` (no back) but explicitly mentioning a "backrest", leading to `chair`; Missing the "wings" of an airplane.

4. **Description: Too Vague**
   - The description is correct but **too generic** or brief to distinguish the GT from other classes. It lacks specific discriminative features.
   - *Example*: "It is a rectangular wooden object." (Could be table, desk, bench, shelf...).

5. **Semantic: Fine-Grained Ambiguity**
   - The GT and Pred classes are **functionally or visually very distinct**. The description is detailed and fits *both* reasonably well, or the distinction is subjective.
   - *Example*: `Desk` vs `Table`; `Vase` vs `Flower Pot`; `Dresser` vs `Night Stand`.

6. **Classifier: Correct Description**
   - The Model Description **clearly and accurately describes the Ground Truth** (perhaps even naming it), but the external classifier assigned the wrong label.
   - *Example*: Description says "This is a piano", but Pred is "Table".

**Output Format:**
Return strict JSON only:
{{
  "category": "Visual: Shape-Similar Confusion" | "Visual: Severe Hallucination" | "Visual: Part Hallucination/Miss" | "Description: Too Vague" | "Semantic: Fine-Grained Ambiguity" | "Classifier: Correct Description",
  "reasoning": "One sentence explaining the choice."
}}
"""

def analyze_sample(sample: Dict) -> Dict:
    prompt = PROMPT_TEMPLATE.format(
        gt_label=sample['ground_truth_label'],
        pred_label=sample['gpt_cls_label'],
        model_output=sample['model_output'].replace('"', '\\"')
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return {**sample, "error_analysis": {"category": "API Error", "reasoning": f"Status {response.status_code}"}}
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError:
            analysis = {"category": "Parse Error", "reasoning": content[:100]}
        
        return {**sample, "error_analysis": analysis}
    except Exception as e:
        print(f"Exception: {e}")
        return {**sample, "error_analysis": {"category": "Exception", "reasoning": str(e)}}

def main():
    parser = argparse.ArgumentParser(description="Analyze PointLLM error cases using an external LLM.")
    parser.add_argument("--input", type=str, default="PointLLM/pointllm_evaluation/ModelNet_classification_prompt0_evaluated_gpt-4.1.json", help="Path to input JSON")
    parser.add_argument("--output", type=str, default="error_analysis_results.json", help="Path to output JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    args = parser.parse_args()

    # Load Data
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    # Filter Errors
    error_cases = [r for r in results if r['ground_truth_label'] != r['gpt_cls_label']]
    print(f"Found {len(error_cases)} error cases.")

    if args.max_samples:
        error_cases = error_cases[:args.max_samples]
        print(f"Processing first {args.max_samples} samples.")

    # Process
    analyzed_results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_sample = {executor.submit(analyze_sample, sample): sample for sample in error_cases}
        
        count = 0
        total = len(error_cases)
        
        for future in as_completed(future_to_sample):
            res = future.result()
            analyzed_results.append(res)
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{total}...")

    # Statistics & Sampling
    stats = {}
    examples = {}

    for res in analyzed_results:
        cat = res.get('error_analysis', {}).get('category', 'Unknown')
        stats[cat] = stats.get(cat, 0) + 1
        
        if cat not in examples:
            examples[cat] = []
        examples[cat].append(res)
    
    print("\n" + "="*50)
    print(" ANALYSIS STATISTICS ")
    print("="*50)
    # Sort by count desc
    for cat, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(analyzed_results)) * 100
        print(f"{cat}: {count} ({percentage:.1f}%)")

    print("\n" + "="*50)
    print(" REPRESENTATIVE EXAMPLES ")
    print("="*50)
    
    for cat in stats.keys():
        if cat == "Unknown" or cat == "API Error" or cat == "Exception":
            continue
            
        print(f"\n--- Category: {cat} ---")
        # Sample up to 2 examples
        cat_examples = examples[cat]
        sample_size = min(2, len(cat_examples))
        sampled = random.sample(cat_examples, sample_size)
        
        for i, ex in enumerate(sampled):
            print(f"\n[Example {i+1}]")
            print(f"GT: {ex['ground_truth_label']} | Pred: {ex['gpt_cls_label']}")
            print(f"Desc: {ex['model_output']}")
            print(f"Reasoning: {ex['error_analysis'].get('reasoning', 'N/A')}")

    # Save
    print(f"\nSaving full results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(analyzed_results, f, indent=2)

if __name__ == "__main__":
    main()
