import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Data loading from results/sae_experiments/ TSVs
# ---------------------------------------------------------------------------

def load_tsv_safe(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, sep="\t")
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return pd.DataFrame()


def collect_t2(folder: str) -> pd.DataFrame:
    """Collect T2 results from linear probes (train-*.tsv where data_path=t2).
    For T2, we trained on 't2' which has held-out slices. We want the accuracy
    on the held-out ('unseen') slices.
    """
    rows = []
    for f in sorted(glob.glob(f"{folder}/train-*.tsv")):
        df = load_tsv_safe(f)
        if df.empty or "accuracy" not in df.columns or "data_path" not in df.columns:
            continue
            
        # Only process t2 linear probe TSVs
        data_path = str(df["data_path"].iloc[0])
        if "t2" not in data_path:
            continue

        if "sae_mode" not in df.columns or "holdout" not in df.columns:
            continue

        mode = df["sae_mode"].iloc[0]
        if pd.isna(mode):
            continue
            
        arch = df["sae_arch"].iloc[0]
        layer = df["sae_layer"].iloc[0]
        seed = int(df["seed"].iloc[0])

        # We care about generalization to the UNSEEN slices.
        unseen_mask = df["holdout"] == "unseen"
        seen_mask = df["holdout"] == "seen"
        
        unseen_acc = df.loc[unseen_mask, "accuracy"].mean() if unseen_mask.any() else np.nan
        seen_acc = df.loc[seen_mask, "accuracy"].mean() if seen_mask.any() else np.nan

        rows.append({
            "mode": mode,
            "arch": arch,
            "layer": layer,
            "seed": seed,
            "t2_unseen_acc": unseen_acc,
            "t2_seen_acc": seen_acc,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_t1(folder: str) -> pd.DataFrame:
    """Collect T1 (grounding) results from sae_t1-*.tsv files.

    Each TSV has per-class rows with an 'accuracy' column.
    We aggregate to mean accuracy per (mode, arch, layer, seed).
    """
    rows = []
    for f in sorted(glob.glob(f"{folder}/sae_t1-*.tsv")):
        df = load_tsv_safe(f)
        if df.empty or "accuracy" not in df.columns:
            continue

        row = {
            "mode": df["sae_mode"].iloc[0],
            "arch": df["sae_arch"].iloc[0],
            "layer": df["sae_layer"].iloc[0],
            "seed": int(df["seed"].iloc[0]),
            "t1_accuracy": df["accuracy"].mean(),
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_causal(folder: str) -> pd.DataFrame:
    """Collect T3 (modular) and T4 (causal) results from sae_causal-*.tsv files.
    """
    rows = []
    for f in sorted(glob.glob(f"{folder}/sae_causal-*.tsv")):
        df = load_tsv_safe(f)
        if df.empty or "accuracy" not in df.columns:
            continue
        if "dataspec_intervene" not in df.columns or "dataspec_downstream" not in df.columns:
            continue

        mode = df["sae_mode"].iloc[0]
        arch = df["sae_arch"].iloc[0]
        layer = df["sae_layer"].iloc[0]
        seed = int(df["seed"].iloc[0])

        # Test 3: Downstream is an atomic concept (not "classes").
        t3_mask = (df["dataspec_downstream"] != "classes") & (df["dataspec_intervene"] != "classes")
        t3_self_mask = t3_mask & (df["dataspec_intervene"] == df["dataspec_downstream"])
        t3_cross_mask = t3_mask & (df["dataspec_intervene"] != df["dataspec_downstream"])

        t3_self_acc = df.loc[t3_self_mask, "accuracy"].mean() if t3_self_mask.any() else np.nan
        t3_cross_acc = df.loc[t3_cross_mask, "accuracy"].mean() if t3_cross_mask.any() else np.nan

        # Test 4: Downstream is "classes". We evaluate prediction of classes when an atomic concept is ablated.
        t4_mask = (df["dataspec_downstream"] == "classes") & (df["dataspec_intervene"] != "classes")
        t4_ablation_accuracy = df.loc[t4_mask, "accuracy"].mean() if t4_mask.any() else np.nan

        rows.append({
            "mode": mode, "arch": arch, "layer": layer, "seed": seed,
            "t3_self_ablation": t3_self_acc,
            "t3_cross_ablation": t3_cross_acc,
            "t4_ablation_accuracy": t4_ablation_accuracy,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def collect_probes(folder: str) -> pd.DataFrame:
    """Collect T1 (grounding) results from sae_t1-*.tsv files.

    Each TSV has per-class rows with an 'accuracy' column.
    We aggregate to mean accuracy per (mode, arch, layer, seed).
    """
    rows = []
    for f in sorted(glob.glob(f"{folder}/train-*.tsv")):
        df = load_tsv_safe(f)
        if df.empty or "accuracy" not in df.columns:
            continue

        row = {
            "mode": df["sae_mode"].iloc[0],
            "arch": df["sae_arch"].iloc[0],
            "layer": df["sae_layer"].iloc[0],
            "seed": int(df["seed"].iloc[0]),
            "probe_accuracy": df["accuracy"].mean(),
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LAYER_ORDER = ["layer1", "layer2", "layer3", "fc"]

HUE_ORDER = [
    "standard_post_hoc", "standard_integrated",
    "topk_post_hoc", "topk_integrated",
]

PALETTE = {
    "standard_post_hoc": "#4878d0",
    "standard_integrated": "#6acc64",
    "topk_post_hoc": "#ee854a",
    "topk_integrated": "#d65f5f",
}


def _bar_plot(df, y_col, title, ylabel, filename, ylim=None):
    """Grouped bar plot: x = layer, hue = arch_mode, error bars = sd over seeds."""
    plot_df = df.copy()
    plot_df["arch_mode"] = plot_df["arch"] + "_" + plot_df["mode"]

    # Filter to canonical layers
    plot_df = plot_df[plot_df["layer"].isin(LAYER_ORDER)]
    plot_df["layer"] = pd.Categorical(plot_df["layer"], categories=LAYER_ORDER, ordered=True)
    plot_df = plot_df.dropna(subset=[y_col]).sort_values("layer")

    if plot_df.empty:
        print(f"  Skipping {filename}: no data for {y_col}")
        return

    # Only include hue levels that exist in the data
    hue_present = [h for h in HUE_ORDER if h in plot_df["arch_mode"].values]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df, x="layer", y=y_col, hue="arch_mode",
        hue_order=hue_present, palette=PALETTE,
        errorbar="sd", order=LAYER_ORDER, capsize=0.05,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("Layer")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(title="Architecture / Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def generate_plots(df):
    os.makedirs("figures_sae", exist_ok=True)

    # T1: Groundedness Ratio
    if "t1_accuracy" in df.columns:
        _bar_plot(
            df, "t1_accuracy",
            "Test 1: Concept Fraction (higher = more grounded)",
            "Ratio", "figures_sae/t1_concept_fraction.png", ylim=(0, 1),
        )

    # T2: Is Token of Type (Unseen slice accuracy)
    if "t2_unseen_acc" in df.columns:
        _bar_plot(
            df, "t2_unseen_acc",
            "Test 2: OOD Probe Accuracy on Unseen Slices (higher = better generalization)",
            "Accuracy", "figures_sae/t2_unseen_acc.png", ylim=(0, 1),
        )

    # General Probes (Baseline classification)
    if "probe_accuracy" in df.columns:
        _bar_plot(
            df, "probe_accuracy",
            "Linear Probe Accuracy on SAE Features",
            "Accuracy", "figures_sae/probe_accuracy.png", ylim=(0, 1),
        )

    # T3: Self-ablation (should be low if modular)
    if "t3_self_ablation" in df.columns:
        _bar_plot(
            df, "t3_self_ablation",
            "Test 3: Self-Ablation Accuracy (lower = more modular)",
            "Accuracy after ablation", "figures_sae/t3_self_ablation.png", ylim=(0, 1),
        )

    # T3: Cross-ablation (should be high if modular)
    if "t3_cross_ablation" in df.columns:
        _bar_plot(
            df, "t3_cross_ablation",
            "Test 3: Cross-Ablation Accuracy (higher = more modular)",
            "Accuracy after ablation", "figures_sae/t3_cross_ablation.png", ylim=(0, 1),
        )

    # T4: Causal ablation
    if "t4_ablation_accuracy" in df.columns:
        _bar_plot(
            df, "t4_ablation_accuracy",
            "Test 4: Class Prediction Accuracy after Concept Ablation",
            "Accuracy after ablation", "figures_sae/t4_ablation_accuracy.png", ylim=(0, 1),
        )

    print("Done generating plots.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    folder = "results/sae_experiments"
    
    print("Collecting T1 results...")
    df_t1 = collect_t1(folder)
    
    print("Collecting T2 (linear probe) results...")
    df_t2 = collect_t2(folder)

    print("Collecting Probe (default) results...")
    df_probes = collect_probes(folder)

    print("Collecting T3/T4 (causal) results...")
    df_causal = collect_causal(folder)

    print("\nMerging results...")
    # Base merge keys
    keys = ["mode", "arch", "layer", "seed"]

    final_df = None
    
    for df_sub, name in [(df_t1, "T1"), (df_t2, "T2"), (df_probes, "Probes"), (df_causal, "Causal")]:
        if df_sub.empty:
            print(f"  No {name} results found.")
            continue
        if final_df is None:
            final_df = df_sub
        else:
            final_df = pd.merge(final_df, df_sub, on=keys, how="outer")

    if final_df is not None and not final_df.empty:
        # Exclude conv_layer0
        final_df = final_df[final_df["layer"].isin(LAYER_ORDER)]

        out_path = "sae_linear_probes_summary.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path} ({len(final_df)} rows)")

        print("\n" + "=" * 70)
        print("Mean results by (mode, arch, layer):")
        print("=" * 70)
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        summary = final_df.groupby(["mode", "arch", "layer"])[numeric_cols].mean()
        # Drop seed from summary display
        if "seed" in summary.columns:
            summary = summary.drop(columns=["seed"])
        print(summary.round(4).to_string())

        print()
        generate_plots(final_df)
    else:
        print("No results found. Are you sure you ran eval_only_sae_parallel.sh?")

if __name__ == "__main__":
    main()
