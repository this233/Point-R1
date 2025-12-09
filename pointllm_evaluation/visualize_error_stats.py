import json
import os
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Define the file path directly
ANALYSIS_FILE = "PointLLM/pointllm_evaluation/analyse.json"

def analyze_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading analysis from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle different json structures if necessary, assuming list of results based on user context
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and 'results' in data: # Fallback if user passed original structure with 'results'
        # If the input was the original evaluation file, we need to check if 'error_analysis' exists
        # But user says they have the "analyzed json", so let's assume it's the output of the previous script
        # which is a list of error objects.
        # However, let's check the first item to be sure.
        raw_results = data['results']
        # If it's the raw file without analysis, we can't process.
        # But let's assume the user meant the output of the analysis script which is usually a list.
        # If the previous script saved `analyzed_results` (a list), then `data` is that list.
        pass
        results = raw_results # This might be wrong if the file is the OUTPUT of the analysis script
    else:
        # If it's the output of the analysis script (list of dicts)
         results = data

    # Filter for items that actually have error analysis
    analyzed_items = [r for r in results if 'error_analysis' in r]
    
    if not analyzed_items:
        print("No analyzed error cases found in the file.")
        return

    total_errors = len(analyzed_items)
    print(f"\nTotal Analyzed Error Cases: {total_errors}")

    # --- 1. Overall Category Distribution ---
    categories = [r['error_analysis'].get('category', 'Unknown') for r in analyzed_items]
    cat_counts = Counter(categories)
    
    print("\n" + "="*60)
    print(" FAILURE CATEGORY DISTRIBUTION ")
    print("="*60)
    df_cat = pd.DataFrame.from_dict(cat_counts, orient='index', columns=['Count'])
    df_cat['Percentage'] = (df_cat['Count'] / total_errors * 100).round(2)
    df_cat = df_cat.sort_values('Count', ascending=False)
    print(df_cat)

    # --- 2. Confusion Matrix per Category ---
    # To see which classes are most affected by which error type
    print("\n" + "="*60)
    print(" TOP CONFUSIONS BY CATEGORY ")
    print("="*60)

    for cat in df_cat.index:
        if cat == 'Unknown': continue
        
        cat_items = [r for r in analyzed_items if r['error_analysis'].get('category') == cat]
        # Count (GT -> Pred) pairs
        confusions = Counter([(r['ground_truth_label'], r['gpt_cls_label']) for r in cat_items])
        
        print(f"\n--- Category: {cat} ({len(cat_items)} cases) ---")
        print(f"Top 5 Class Confusions (GT -> Pred):")
        for (gt, pred), count in confusions.most_common(5):
            print(f"  {gt} -> {pred}: {count}")

    # --- 3. Representative Examples (One per top confusion) ---
    print("\n" + "="*60)
    print(" REPRESENTATIVE EXAMPLES FOR TOP ISSUES ")
    print("="*60)
    
    # Get top 3 categories
    top_cats = df_cat.index[:3]
    
    for cat in top_cats:
        cat_items = [r for r in analyzed_items if r['error_analysis'].get('category') == cat]
        if not cat_items: continue
        
        # Get the most common confusion pair for this category
        most_common_conf = Counter([(r['ground_truth_label'], r['gpt_cls_label']) for r in cat_items]).most_common(1)
        if not most_common_conf: continue
        
        top_gt, top_pred = most_common_conf[0][0]
        
        # Find an example of this specific confusion
        example = next((r for r in cat_items if r['ground_truth_label'] == top_gt and r['gpt_cls_label'] == top_pred), None)
        
        if example:
            print(f"\n>>> Category: {cat}")
            print(f"Typical Pattern: {top_gt} -> {top_pred}")
            print(f"Model Output: \"{example['model_output']}\"")
            print(f"Reasoning: {example['error_analysis'].get('reasoning')}")

if __name__ == "__main__":
    analyze_results(ANALYSIS_FILE)

