import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/results/results/"
SAE_CSV = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/sae_summary.csv"
OUT_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/code/gaila/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Original Baseline Results (RawCnn)
t1_df = pd.read_csv(os.path.join(RESULTS_DIR, 't1.tsv.zip'), sep='\t')
t2_df = pd.read_csv(os.path.join(RESULTS_DIR, 't2.tsv.zip'), sep='\t')
t3_df = pd.read_csv(os.path.join(RESULTS_DIR, 't3.tsv.zip'), sep='\t')
t4_df = pd.read_csv(os.path.join(RESULTS_DIR, 't4.tsv.zip'), sep='\t')

baseline_rawcnn = {
    't1_acc': t1_df[t1_df['pretrain_model'] == 'RawCnn']['accuracy'].mean(),
    't2_acc': t2_df[t2_df['pretrain_model'] == 'RawCnn']['accuracy'].mean(),
    't3_ablated': t3_df[(t3_df['pretrain_model'] == 'RawCnn') & (t3_df['condition'] == 'ablated')]['accuracy'].mean(),
    't3_others': t3_df[(t3_df['pretrain_model'] == 'RawCnn') & (t3_df['condition'] == 'others')]['accuracy'].mean(),
    't4_ablated': t4_df[(t4_df['pretrain_model'] == 'RawCnn') & (t4_df['condition'] == 'ablated')]['accuracy'].mean(),
    't4_others': t4_df[(t4_df['pretrain_model'] == 'RawCnn') & (t4_df['condition'] == 'others')]['accuracy'].mean(),
}

# 2. Load SAE Results (Post-hoc Standard and ToolK, fc layer)
sae_df = pd.read_csv(SAE_CSV)
sae_fc = sae_df[sae_df['layer'] == 'fc']

std_fc = sae_fc[sae_fc['arch'] == 'standard']
topk_fc = sae_fc[sae_fc['arch'] == 'topk']

models = ['Original (RawCnn)', 'Post-Hoc Standard', 'Post-Hoc TopK (k=32)']

def plot_bar(fig_num, title, models, values, ylabel, filename, is_grouped=False, group_labels=None, ymin=0.0):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if is_grouped:
        x = np.arange(len(models))
        width = 0.35
        # values should be a list of lists, [vals_group1, vals_group2]
        rects1 = ax.bar(x - width/2, values[0], width, label=group_labels[0], color='skyblue', edgecolor='black')
        rects2 = ax.bar(x + width/2, values[1], width, label=group_labels[1], color='salmon', edgecolor='black')
        ax.legend()
    else:
        x = np.arange(len(models))
        ax.bar(x, values, color=['lightgrey', 'skyblue', 'salmon'], edgecolor='black', width=0.6)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    
    # Optional y-limit setting, for diffs we leave it auto if it has diff metrics
    max_val = max(values)
    if max_val <= 1.0 and ymin == 0.0:
        ax.set_ylim([ymin, 1.05])
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

# Figure 3: is_grounded (Accuracy vs F1 Seen)
fig3_vals = [
    baseline_rawcnn['t1_acc'],
    std_fc['avg_f1_seen'].mean(),
    topk_fc['avg_f1_seen'].mean()
]
plot_bar(3, 'Figure 3: Test 1 (Groundedness)', models, fig3_vals, 'Accuracy (Original) / F1 Seen (SAE)', 'fig3.png')

# Figure 4: is_token_of_type (Accuracy un-seen vs F1 unseen)
fig4_vals = [
    baseline_rawcnn['t2_acc'],
    std_fc['avg_f1_unseen'].mean(),
    topk_fc['avg_f1_unseen'].mean()
]
plot_bar(4, 'Figure 4: Test 2 (Token of Type / Unseen)', models, fig4_vals, 'Accuracy (Original) / F1 Unseen (SAE)', 'fig4.png')

# Figure 5: is_modular
delta_baseline = baseline_rawcnn['t3_others'] - baseline_rawcnn['t3_ablated']
fig5_vals = [
    delta_baseline,
    std_fc['modularity_ratio'].mean(),
    topk_fc['modularity_ratio'].mean()
]
plot_bar(5, 'Figure 5: Test 3 (Modularity)', models, fig5_vals, 'Diff (Others-Ablated) for Orig / Modularity Ratio for SAE', 'fig5.png')

# Figure 6: is_causal
delta_baseline_t4 = baseline_rawcnn['t4_others'] - baseline_rawcnn['t4_ablated']
std_t4_gap = std_fc['t4_causal_gap'].mean() if 't4_causal_gap' in std_fc.columns and std_fc['t4_causal_gap'].notna().any() else 0.0
topk_t4_gap = topk_fc['t4_causal_gap'].mean() if 't4_causal_gap' in topk_fc.columns and topk_fc['t4_causal_gap'].notna().any() else 0.0
fig6_vals = [
    delta_baseline_t4,
    std_t4_gap,
    topk_t4_gap,
]
plot_bar(6, 'Figure 6: Test 4 (Causality)', models, fig6_vals, 'Causal Gap (Others Acc - Ablated Acc)', 'fig6.png', ymin=0.0)

print("Charts successfully saved to:", OUT_DIR)
