import os
import json
import pandas as pd
import glob
import re

def parse_filename(filename):
    """
    Extracts metadata from filenames like:
    sae_mode=post_hoc_arch=standard_feat=4096_layer=conv_layer0_model=RawCnn_data=default_seed=10_l1=0.001_eval.json
    """
    metadata = {}
    # Remove extension and trailing tags
    name = filename.replace("_eval.json", "").replace("_concept_graph.json", "")
    
    # Split by underscores. Note that some values might contain underscores (like layer names)
    # The structure is key1=val1_key2=val2...
    # But wait, looking at the name: sae_mode=post_hoc_arch=standard...
    # 'sae_mode=post' is split? No, it's 'post_hoc'.
    # Actually, let's try splitting by underscores and then checking if it contains '='
    
    parts = name.split("_")
    current_key = None
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            metadata[k] = v
            current_key = k
        elif current_key:
            # Append to previous value (for cases like post_hoc or conv_layer0)
            metadata[current_key] += "_" + p
            
    return metadata

def aggregate_results(results_dir):
    eval_files = glob.glob(os.path.join(results_dir, "*_eval.json"))
    
    all_data = []
    
    for f_path in eval_files:
        f_name = os.path.basename(f_path)
        meta = parse_filename(f_name)
        
        with open(f_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {f_path}")
                continue
        
        # 1. Concept Assignment (Seen F1)
        assignment = data.get("concept_assignment", {})
        f1_seen_vals = [v.get("f1_seen", 0) for v in assignment.values()]
        avg_f1_seen = sum(f1_seen_vals) / len(f1_seen_vals) if f1_seen_vals else 0
        
        # 2. Test 2 (Unseen F1)
        test2 = data.get("test2_is_token_of_type", {})
        f1_unseen_vals = [v.get("f1_unseen", 0) for v in test2.values() if isinstance(v, dict)]
        avg_f1_unseen = sum(f1_unseen_vals) / len(f1_unseen_vals) if f1_unseen_vals else 0
        
        # 3. Test 3 (Modularity)
        test3 = data.get("test3_is_modular", {})
        delta_on_vals = [v.get("delta_on", 0) for v in test3.values() if isinstance(v, dict)]
        delta_off_vals = [v.get("delta_off", 0) for v in test3.values() if isinstance(v, dict)]
        delta_other_vals = [v.get("delta_other_abs_mean", 0) for v in test3.values() if isinstance(v, dict)]
        
        avg_delta_on = sum(delta_on_vals) / len(delta_on_vals) if delta_on_vals else 0
        avg_delta_off = sum(delta_off_vals) / len(delta_off_vals) if delta_off_vals else 0
        avg_delta_other = sum(delta_other_vals) / len(delta_other_vals) if delta_other_vals else 0
        
        # Modularity Score: (on - off) / (on + |off| + other) -- simple heuristic
        denom = (abs(avg_delta_on) + abs(avg_delta_off) + avg_delta_other)
        modularity_ratio = (avg_delta_on - avg_delta_off) / denom if denom > 1e-9 else 0

        row = {
            **meta,
            "avg_f1_seen": avg_f1_seen,
            "avg_f1_unseen": avg_f1_unseen,
            "avg_delta_on": avg_delta_on,
            "avg_delta_off": avg_delta_off,
            "avg_delta_other": avg_delta_other,
            "modularity_ratio": modularity_ratio,
        }
        all_data.append(row)
        
    df = pd.DataFrame(all_data)
    return df

if __name__ == "__main__":
    RESULTS_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/code/gaila/results_sae/sae_experiments"
    output_df = aggregate_results(RESULTS_DIR)
    
    if not output_df.empty:
        # Sort for readability
        sort_cols = [c for c in ["arch", "layer", "seed"] if c in output_df.columns]
        output_df = output_df.sort_values(by=sort_cols)
        
        output_path = "sae_summary.csv"
        output_df.to_csv(output_path, index=False)
        print(f"Summary saved to {output_path}")
        
        # Print a quick summary grouped by arch and layer
        summary = output_df.groupby(["arch", "layer"])[["avg_f1_seen", "avg_f1_unseen", "modularity_ratio"]].mean()
        print("\nMean scores across seeds:")
        print(summary)
    else:
        print("No data found.")
