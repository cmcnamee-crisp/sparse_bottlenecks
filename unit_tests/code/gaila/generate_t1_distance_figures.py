import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

SAE_CSV = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/sae_summary.csv"
OUT_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/code/gaila/figures_layers"

os.makedirs(OUT_DIR, exist_ok=True)

# 2. Load SAE Results
sae_df = pd.read_csv(SAE_CSV)
std_df = sae_df[sae_df['arch'] == 'standard']
topk_df = sae_df[sae_df['arch'] == 'topk']

layer_order = ['conv_layer0', 'layer1', 'layer2', 'layer3', 'fc']

# Prepare data arrays in the specified layer_order
def extract_metric(df, metric):
    res = []
    for l in layer_order:
        val = df[df['layer'] == l][metric].mean()
        res.append(val if pd.notnull(val) else 0.0)
    return res

std_total_dist = extract_metric(std_df, 'mean_total_sparse_code_dist')
topk_total_dist = extract_metric(topk_df, 'mean_total_sparse_code_dist')

std_concept_dist = extract_metric(std_df, 'mean_concept_neuron_dist')
topk_concept_dist = extract_metric(topk_df, 'mean_concept_neuron_dist')

std_bg_frac = extract_metric(std_df, 'background_fraction')
topk_bg_frac = extract_metric(topk_df, 'background_fraction')

# General Plotting Function for Lines
def plot_layers(fig_num, title, ylabel, series_dict, filename, is_fraction=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # plot each series
    x = np.arange(len(layer_order))
    colors = ['skyblue', 'salmon', 'green', 'purple']
    markers = ['s', '^', 'D', 'x']
    
    i = 0
    for label, vals in series_dict.items():
        if all(v == 0.0 for v in vals): 
            pass
        ax.plot(x, vals, color=colors[i], marker=markers[i], linewidth=2.5, markersize=8, label=label)
        i += 1

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_order, fontsize=11, rotation=15)
    
    if is_fraction:
        ax.set_ylim([0.0, 1.05])
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

# Figure A: Total Distance
plot_layers(
    "A", 
    'Test 1: Total Sparse Code Distance', 
    'L1 Distance', 
    {
        'Post-Hoc Standard SAE': std_total_dist,
        'Post-Hoc TopK (k=32) SAE': topk_total_dist
    },
    't1_total_dist.png',
)

# Figure B: Concept Distance
plot_layers(
    "B", 
    'Test 1: Concept Neuron Distance', 
    'L1 Distance', 
    {
        'Post-Hoc Standard SAE': std_concept_dist,
        'Post-Hoc TopK (k=32) SAE': topk_concept_dist
    },
    't1_concept_dist.png'
)

# Figure C: Background Fraction
plot_layers(
    "C", 
    'Test 1: Background Fraction', 
    'Fraction of Variance NOT Explained by Concept', 
    {
        'Post-Hoc Standard SAE': std_bg_frac,
        'Post-Hoc TopK (k=32) SAE': topk_bg_frac
    },
    't1_background_fraction.png',
    is_fraction=True
)

print("Layer-wise distance charts successfully saved to:", OUT_DIR)
