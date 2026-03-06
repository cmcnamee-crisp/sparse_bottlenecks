import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/results/results/"
SAE_CSV = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/sae_summary.csv"
OUT_DIR = "/Users/cameronmcnamee/Desktop/sparse_bottlenecks/unit_tests/code/gaila/figures_layers"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Original Baseline Results (RawCnn) for Layers
layers_df = pd.read_csv(os.path.join(RESULTS_DIR, 'layers.tsv.zip'), sep='\t')
rawcnn_layers = layers_df[layers_df['pretrain_model'] == 'RawCnn']

# The paper evaluated 'target'='probing' (concepts) and 'downstream' (full class logic).
# Test 1 corresponds to probing the concepts accurately overall. 
# layers.tsv has 'holdout_many'. For basic concept probing (Test 1), accuracy is usually overall, we'll take mean.
base_probing = rawcnn_layers[(rawcnn_layers['target'] == 'probing') & (rawcnn_layers['holdout_many'] == False)].groupby('layer')['value'].mean().to_dict()

# Test 2 is unseen generalization. The paper's Figure 4 typically showcases the 'N-1 Slices' setting 
# which corresponds to holdout_many=False (we hold out 1 slice, train on the rest).
base_probing_unseen = rawcnn_layers[(rawcnn_layers['target'] == 'probing') & (rawcnn_layers['holdout_many'] == False)].groupby('layer')['value'].mean().to_dict()


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

std_f1_seen = extract_metric(std_df, 'avg_f1_seen')
topk_f1_seen = extract_metric(topk_df, 'avg_f1_seen')
base_acc_seen = [base_probing.get(l, 0.0) for l in layer_order]

std_f1_unseen = extract_metric(std_df, 'avg_f1_unseen')
topk_f1_unseen = extract_metric(topk_df, 'avg_f1_unseen')
base_acc_unseen = [base_probing_unseen.get(l, 0.0) for l in layer_order]

std_modularity = extract_metric(std_df, 'modularity_ratio')
topk_modularity = extract_metric(topk_df, 'modularity_ratio')

std_t4_gap = extract_metric(std_df, 't4_causal_gap')
topk_t4_gap = extract_metric(topk_df, 't4_causal_gap')


# For Modularity, the layer-wise baseline isn't stored in layers.tsv, but we do have it in t3.tsv.zip? wait no, t3.tsv.zip only has fc. The paper's dataset only measured ablations on the very end. The SAEs were trained natively per layer so we can show those layered, but base will just be a point or flat line at FC? Or omit base if we don't have it layered. Let's just plot SAEs for test 3.


# General Plotting Function for Lines
def plot_layers(fig_num, title, ylabel, series_dict, filename, ymin=0.0, ymax=1.05):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # plot each series
    x = np.arange(len(layer_order))
    colors = ['black', 'skyblue', 'salmon', 'green', 'purple']
    markers = ['o', 's', '^', 'D', 'x']
    
    i = 0
    for label, vals in series_dict.items():
        if all(v == 0.0 for v in vals) and label != 'Original Base Probing (RawCnn)': 
            pass # Skip all zeros just in case
        ax.plot(x, vals, color=colors[i], marker=markers[i], linewidth=2.5, markersize=8, label=label)
        i += 1

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_order, fontsize=11, rotation=15)
    
    if ymin is not None and ymax is not None:
        ax.set_ylim([ymin, ymax])
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()

# Figure 3
plot_layers(
    3, 
    'Figure 3: Test 1 (Groundedness) Across Layers', 
    'Accuracy (Original) / F1 Seen (SAE)', 
    {
        'Linear Probes (RawCnn)': base_acc_seen,
        'Post-Hoc Standard SAE': std_f1_seen,
        'Post-Hoc TopK (k=32) SAE': topk_f1_seen
    },
    'fig3_layers.png'
)

# Figure 4
plot_layers(
    4,
    'Figure 4: Test 2 (Token of Type / Unseen) Across Layers',
    'Accuracy Unseen (Original) / F1 Unseen (SAE)',
    {
        'Linear Probes (RawCnn)': base_acc_unseen,
        'Post-Hoc Standard SAE': std_f1_unseen,
        'Post-Hoc TopK (k=32) SAE': topk_f1_unseen
    },
    'fig4_layers.png'
)

# Figure 5
plot_layers(
    5,
    'Figure 5: Test 3 (Modularity) Across Layers',
    'Modularity Ratio',
    {
        'Post-Hoc Standard SAE': std_modularity,
        'Post-Hoc TopK (k=32) SAE': topk_modularity
    },
    'fig5_layers.png'
)

# Figure 6
plot_layers(
    6,
    'Figure 6: Test 4 (Causality) Across Layers',
    'Causal Gap (Others Acc - Ablated Acc)',
    {
        'Post-Hoc Standard SAE': std_t4_gap,
        'Post-Hoc TopK (k=32) SAE': topk_t4_gap
    },
    'fig6_layers.png',
    ymin=None,
    ymax=None,
)

print("Layer-wise charts successfully saved to:", OUT_DIR)
