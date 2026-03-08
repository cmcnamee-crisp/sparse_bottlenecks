"""SAE Evaluation Script.

Runs all four Fodor-criteria unit tests using a trained SAE in place of
linear probes and INLP.

Evaluation flow
---------------
0. Prerequisite — Concept Assignment
   Compute F1 scores for every SAE neuron against every concept value
   (3 layouts, 3 shapes, 2 strokes = 8 values total).
   Saves a bipartite graph (as a JSON adjacency list + edge weights)
   for downstream analysis.  The neuron with the highest F1 for any
   given concept is "assigned" to that concept.

1. Test 1 — is_grounded
   On counterfactual minimal pairs (T1 dataset), compare:
     (a) Total sparse code distance  ||A1 - A2|| for each pair.
     (b) Concept-neuron distance     ||A1_C - A2_C|| (assigned neurons only).
   Ratio (b)/(a) quantifies how much variance the SAE attributes to concept
   vs. background nuisance.  Also reports predict accuracy with z_hat.

2. Test 2 — is_token_of_type
   Evaluate the F1 of assigned neurons on UNSEEN class slices.
   High F1 = neuron generalizes across compositional contexts.

3. Test 3 — is_modular
   For each minimal pair (x1→x2, concept changes from v1 to v2):
     - Compute activations A1, A2.
     - Neurons expected OFF: assigned to v1 should decrease.
     - Neurons expected ON : assigned to v2 should increase.
     - Neurons expected SAME: assigned to other concepts should not change.
   Reports mean activation delta per expected-status group.

4. Test 4 — is_causal
   For each sample, zero the assigned neuron for one concept →
   decode → predict.  Measure:
     (a) Accuracy on the ablated concept dimension (should → random).
     (b) Accuracy on the other concept dimensions (should remain high).
"""

import os
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.metrics import f1_score

import flags
import helpers
import specifications
from sae_train import get_sae_flags, load_checkpoint, sae_id


os.makedirs("results_sae", exist_ok=True)


# ---------------------------------------------------------------------------
# Utility: gather all latents + concept labels
# ---------------------------------------------------------------------------

CONCEPT_DIMS = ["layout", "shape", "stroke"]

# Baseline configuration
N_BASELINE_PERMUTATIONS = 100   # For F1 permutation test (cheap: numpy only)
N_BASELINE_RANDOM_SAMPLES = 100 # For random neuron baselines (cheap: numpy only)
N_BASELINE_CAUSAL_SAMPLES = 20  # For Test 4 random ablation (expensive: decode + forward)
ACTIVATION_EPSILON = 1e-3       # Activations below this are zeroed for distance metrics


def _fast_f1_all_neurons(y_true: np.ndarray, A_binary: np.ndarray) -> np.ndarray:
    """Vectorized F1 for all neurons at once.

    Args:
        y_true: (N,) binary int array for one concept value.
        A_binary: (N, num_features) binary activation matrix.

    Returns:
        (num_features,) F1 score per neuron.
    """
    y = y_true[:, None].astype(float)
    tp = (y * A_binary).sum(axis=0)
    fp = ((1 - y) * A_binary).sum(axis=0)
    fn = (y * (1 - A_binary)).sum(axis=0)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0)

def _get_concept_labels(datadesc: pd.DataFrame, FLAGS) -> Dict[str, np.ndarray]:
    """Return dict mapping concept_name -> integer label array (one per sample)."""
    concept_labels = {}
    for dim in CONCEPT_DIMS:
        old_spec = FLAGS.dataspec
        FLAGS.dataspec = dim
        _, dspec = helpers.load_info(FLAGS)
        FLAGS.dataspec = old_spec
        arr = datadesc.classname.map(
            {cn: v["classlabel"] for cn, v in dspec.items()}
        ).values
        concept_labels[dim] = arr.astype(int)
    return concept_labels


def _get_concept_value_names(datadesc: pd.DataFrame, FLAGS) -> Dict[str, List[str]]:
    """Return dict mapping concept_dim -> List[concept_value_name]."""
    value_names = {}
    for dim in CONCEPT_DIMS:
        old_spec = FLAGS.dataspec
        FLAGS.dataspec = dim
        _, dspec = helpers.load_info(FLAGS)
        FLAGS.dataspec = old_spec
        # Invert: label -> name
        label_to_name = {}
        for cn, v in dspec.items():
            label_to_name[v["classlabel"]] = v["conceptname"]
        ordered = [label_to_name[i] for i in sorted(label_to_name)]
        value_names[dim] = ordered
    return value_names


def _load_all_latents_and_labels(FLAGS, layer: str, device: str):
    """Load pre-computed latents + all concept labels for the full dataset."""
    FLAGS.dataspec = "classes"
    datadesc, dataspec = helpers.load_info(FLAGS)
    dataset_path = flags.transformed_id(FLAGS, is_raw=True)

    latents = helpers._get_vector_layer(dataset_path, device, layer)
    if FLAGS.sae_mode == "integrated":
        latents = latents.flatten(1)
    class_labels = th.tensor(datadesc.classlabel.values, dtype=th.long, device=device)
    concept_labels = _get_concept_labels(datadesc, FLAGS)  # each is np.ndarray
    value_names = _get_concept_value_names(datadesc, FLAGS)
    is_test = datadesc.test.values.astype(bool)

    # Seen/unseen masks (holdout mask = "seen classes"; complement = "unseen" for OOD test)
    # We need per-dim seen/unseen info; use the layout holdout as a proxy for now.
    # Each concept dim has its own holdout definition in the original paper.
    # We approximate "unseen" as: test set AND classname is marked holdout in *any* dim.
    holdout_any = np.zeros(len(datadesc), dtype=bool)
    for dim in CONCEPT_DIMS:
        old_spec = FLAGS.dataspec
        FLAGS.dataspec = dim
        _, dspec = helpers.load_info(FLAGS)
        FLAGS.dataspec = old_spec
        holdout_mask_dim = datadesc.classname.map(
            {cn: v["holdout"] for cn, v in dspec.items()}
        ).values.astype(bool)
        holdout_any |= holdout_mask_dim

    is_unseen = holdout_any  # these were held out during SAE training

    return latents, class_labels, concept_labels, value_names, datadesc, dataspec, is_test, is_unseen


# ---------------------------------------------------------------------------
# 0. Prerequisite — Concept Assignment
# ---------------------------------------------------------------------------

def compute_concept_assignment(
    sae,
    latents: th.Tensor,
    concept_labels: Dict[str, np.ndarray],
    value_names: Dict[str, List[str]],
    sae_mode: str,
    activation_threshold: float = 0.0,
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[Dict[str, int], Dict[str, float], Dict]:
    """Compute per-neuron F1 scores against each concept value.

    Returns
    -------
    assignment : dict  concept_value_str -> neuron_idx (best neuron)
    f1_scores  : dict  concept_value_str -> f1 score of assigned neuron
    graph      : bipartite graph dict suitable for JSON serialisation
                 {"nodes": [...], "edges": [...]}
    """
    print("[Prerequisite] Computing SAE activations ...")
    sae.eval()
    all_acts = []
    
    # We must flatten integrated SAE inputs to match training
    is_integrated = (sae_mode == "integrated")
    
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
            if is_integrated:
                z_batch = z_batch.flatten(1)
            a = sae.encode(z_batch)
            all_acts.append(a.cpu().numpy())
    A = np.vstack(all_acts)  # (N, num_features)
    A_binary = (A > activation_threshold).astype(int)

    num_features = A.shape[1]
    print(f"[Prerequisite] Activations shape: {A.shape}")

    # Build full F1 table: concept_value -> array of F1 scores (one per neuron)
    concept_keys = []
    f1_table = {}  # concept_key -> np.ndarray of shape (num_features,)

    for dim, labels_arr in concept_labels.items():
        n_values = len(value_names[dim])
        for v_idx in range(n_values):
            y_true = (labels_arr == v_idx).astype(int)
            concept_key = f"{dim}:{value_names[dim][v_idx]}"
            concept_keys.append(concept_key)
            f1_arr = _fast_f1_all_neurons(y_true, A_binary)
            f1_table[concept_key] = f1_arr

    print("[Prerequisite] F1 table computed.")

    # Assignment: for each concept value, pick the best neuron.
    assignment = {}  # concept_key -> neuron_idx
    f1_scores = {}   # concept_key -> best F1

    for ck, f1_arr in f1_table.items():
        best_neuron = int(np.argmax(f1_arr))
        assignment[ck] = best_neuron
        f1_scores[ck] = float(f1_arr[best_neuron])
        print(f"  {ck:30s} -> neuron {best_neuron:4d}  (F1={f1_arr[best_neuron]:.4f})")

    # Permutation baseline: shuffle concept labels, recompute max-F1.
    print(f"[Prerequisite] Permutation baseline ({N_BASELINE_PERMUTATIONS} perms) ...")
    null_max_f1 = {ck: [] for ck in concept_keys}
    for _perm in range(N_BASELINE_PERMUTATIONS):
        for dim, labels_arr in concept_labels.items():
            shuffled = np.random.permutation(labels_arr)
            for v_idx in range(len(value_names[dim])):
                y_perm = (shuffled == v_idx).astype(int)
                ck = f"{dim}:{value_names[dim][v_idx]}"
                f1_perm = _fast_f1_all_neurons(y_perm, A_binary)
                null_max_f1[ck].append(float(f1_perm.max()))

    baseline_f1 = {}
    for ck in concept_keys:
        nm = float(np.mean(null_max_f1[ck]))
        ns = float(np.std(null_max_f1[ck]))
        baseline_f1[ck] = {"null_mean": nm, "null_std": ns}
        print(f"  {ck:30s} | real={f1_scores[ck]:.4f} | null={nm:.4f}±{ns:.4f}")

    # Build bipartite graph for downstream analysis.
    edges = []
    for ck, f1_arr in f1_table.items():
        for neuron_idx in range(num_features):
            w = float(f1_arr[neuron_idx])
            if w > 0.0:
                edges.append({
                    "concept": ck,
                    "neuron": neuron_idx,
                    "f1": w,
                    "assigned": (assignment[ck] == neuron_idx),
                })
    graph = {
        "nodes_concepts": concept_keys,
        "nodes_neurons": list(range(num_features)),
        "edges": edges,
        "assignment": assignment,
        "f1_scores": f1_scores,
        "baseline_f1": baseline_f1,
    }
    return assignment, f1_scores, graph


# ---------------------------------------------------------------------------
# 1. Test 1 — is_grounded
# ---------------------------------------------------------------------------

def test_is_grounded(
    sae,
    backbone,  # The full CNN model used to predict the final logits.
    sae_layer: str,
    latents_t1: th.Tensor,
    datadesc_t1: pd.DataFrame,
    assignment: Dict[str, int],
    class_labels_t1: th.Tensor,
    device: str,
    batch_size: int = 512,
) -> Dict:
    """Test 1: is_grounded — counterfactual minimal pairs.

    Expects t1 dataset (datadesc_t1) where rows with the same background
    parameters differ only in their concept label.

    For each pair we measure:
      - total_dist    : L1 distance of full sparse codes
      - concept_dist  : L1 distance restricted to concept-assigned neurons
      - background_dist = total_dist - concept_dist (proxy for background sensitivity)

    Also evaluates classification accuracy using z_hat (reconstructed latents).
    """
    print("[Test 1] Computing T1 activations ...")
    sae.eval()
    all_acts = []
    all_z_hat = []
    with th.no_grad():
        for i in range(0, len(latents_t1), batch_size):
            z_batch = latents_t1[i : i + batch_size].float().to(device)
            z_hat, a, _ = sae(z_batch)
            all_acts.append(a.cpu().numpy())
            all_z_hat.append(z_hat.cpu())
    A = np.vstack(all_acts)
    Z_hat = th.cat(all_z_hat, dim=0)

    # Zero out near-zero activations to prevent many tiny values
    # from aggregating into misleading distance measures.
    A[np.abs(A) < ACTIVATION_EPSILON] = 0.0
    print(f"  [Test 1] Applied epsilon={ACTIVATION_EPSILON} threshold; "
          f"sparsity={100 * (A == 0).mean():.1f}%")

    concept_neuron_indices = sorted(set(assignment.values()))

    # Counterfactual pairs: group by background_id if available, otherwise
    # pair all samples that share the same background param hash.
    # For now use classname-agnostic pairing via group_id column if present;
    # otherwise skip pair analysis and report marginal distances.
    results = {}

    if "group_id" not in datadesc_t1.columns:
        # Reconstruct group_id: images with same background are usually sequential or 
        # can be paired by their original image_id if it's consistent.
        # In T1, pairs are often (i, i+offset). Let's try to find pairs by the number of unique classes.
        # For simplicity, if they aren't provided, we try to use image_id % (num_images_in_t1 / 2).
        num_classes_in_t1 = len(datadesc_t1.classname.unique())
        num_images = len(datadesc_t1)
        # Based on typical T1 structure, if it's 2 samples per background:
        if num_images % 2 == 0:
            datadesc_t1["group_id"] = np.arange(num_images) % (num_images // 2)
            print(f"  [Test 1] Reconstructed group_id assuming {num_images//2} pairs.")

    if "group_id" in datadesc_t1.columns:
        ids = datadesc_t1.group_id.values
        unique_ids = np.unique(ids)
        # Pre-compute all pair diffs for efficient baseline computation.
        all_pair_diffs = []
        for gid in unique_ids:
            idx_in_group = np.where(ids == gid)[0]
            if len(idx_in_group) < 2:
                continue
            A_group = A[idx_in_group]
            for i in range(len(A_group)):
                for j in range(i + 1, len(A_group)):
                    all_pair_diffs.append(np.abs(A_group[i] - A_group[j]))

        if all_pair_diffs:
            all_pair_diffs = np.stack(all_pair_diffs)  # (num_pairs, num_features)
            total_dists = all_pair_diffs.sum(axis=1)
            concept_dists = all_pair_diffs[:, concept_neuron_indices].sum(axis=1)

            mean_total = float(total_dists.mean())
            mean_concept = float(concept_dists.mean())
            results["mean_total_sparse_code_dist"] = mean_total
            results["mean_concept_neuron_dist"] = mean_concept
            results["mean_background_dist"] = mean_total - mean_concept
            if mean_total > 0:
                results["concept_neuron_fraction"] = mean_concept / mean_total
                results["background_fraction"] = (mean_total - mean_concept) / mean_total

            # Random neuron baseline: sample random subsets of same size.
            n_concept = len(concept_neuron_indices)
            rand_fracs = []
            for _ in range(N_BASELINE_RANDOM_SAMPLES):
                rand_idx = np.random.choice(A.shape[1], size=n_concept, replace=False)
                rand_dists = all_pair_diffs[:, rand_idx].sum(axis=1)
                rand_fracs.append(float(rand_dists.mean() / mean_total) if mean_total > 0 else 0.0)
            results["baseline_random_neuron_frac_mean"] = float(np.mean(rand_fracs))
            results["baseline_random_neuron_frac_std"] = float(np.std(rand_fracs))
            print(f"  [Test 1] Concept neuron fraction: {results.get('concept_neuron_fraction', 'N/A'):.4f}")
            print(f"  [Test 1] Random baseline fraction: {results['baseline_random_neuron_frac_mean']:.4f} ± {results['baseline_random_neuron_frac_std']:.4f}")
        else:
            results["note"] = "no valid pairs found"
    else:
        # No pair ids available; report marginal activation stats instead.
        results["note"] = "no group_id column; skipping pairwise analysis"
        results["mean_activation_l2"] = float(np.mean(np.linalg.norm(A, axis=1)))

    # Classification accuracy using z_hat (reconstucted latents).
    if backbone is not None:
        backbone.eval()
        with th.no_grad():
            logits = backbone.forward_from_layer(Z_hat.to(device), sae_layer)
        preds = logits.argmax(dim=-1).cpu()
        labels_cpu = class_labels_t1.cpu()
        acc = (preds == labels_cpu).float().mean().item()
        results["classify_accuracy_with_zhat"] = acc
        print(f"  [Test 1] Classification accuracy via z_hat: {acc:.4f}")

    print(f"  [Test 1] Results: {results}")
    return results


# ---------------------------------------------------------------------------
# 2. Test 2 — is_token_of_type
# ---------------------------------------------------------------------------

def test_is_token_of_type(
    sae,
    latents: th.Tensor,
    concept_labels: Dict[str, np.ndarray],
    value_names: Dict[str, List[str]],
    assignment: Dict[str, int],
    is_unseen: np.ndarray,
    sae_mode: str,
    activation_threshold: float = 0.0,
    device: str = "cpu",
    batch_size: int = 512,
) -> Dict:
    """Test 2: is_token_of_type — OOD generalization of concept neurons.

    The SAE was trained on "seen" slices.  This test evaluates how well
    the assigned neurons fire on "unseen" compositions (holdout classes).
    """
    print("[Test 2] Computing activations on unseen samples ...")
    sae.eval()
    all_acts = []
    is_integrated = (sae_mode == "integrated")
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
            if is_integrated:
                z_batch = z_batch.flatten(1)
            a = sae.encode(z_batch)
            all_acts.append(a.cpu().numpy())
    A = np.vstack(all_acts)
    A_binary = (A > activation_threshold).astype(int)

    unseen_mask = is_unseen  # bool array over all samples

    results = {}
    print(f"  [Test 2] Unseen samples: {unseen_mask.sum()} / {len(unseen_mask)}")

    for dim in CONCEPT_DIMS:
        labels_arr = concept_labels[dim]
        for v_idx, vname in enumerate(value_names[dim]):
            concept_key = f"{dim}:{vname}"
            neuron_idx = assignment[concept_key]
            y_true = (labels_arr[unseen_mask] == v_idx).astype(int)
            y_pred = A_binary[unseen_mask, neuron_idx]
            f1 = f1_score(y_true, y_pred, zero_division=0)
            results[concept_key] = {"f1_unseen": f1, "neuron": neuron_idx}
            print(f"  {concept_key:30s} -> F1 (unseen) = {f1:.4f}")

    return results


# ---------------------------------------------------------------------------
# 3. Test 3 — is_modular
# ---------------------------------------------------------------------------

def test_is_modular(
    sae,
    latents: th.Tensor,
    concept_labels: Dict[str, np.ndarray],
    value_names: Dict[str, List[str]],
    assignment: Dict[str, int],
    datadesc: pd.DataFrame,
    sae_mode: str,
    device: str = "cpu",
    batch_size: int = 512,
) -> Dict:
    """Test 3: is_modular — input intervention analysis.

    For each concept dimension, we pair minimal pairs (same background,
    different concept value) and check:
      - Neurons assigned to the REMOVED concept decrease (turn off).
      - Neurons assigned to the ADDED  concept increase (turn on).
      - Neurons assigned to OTHER concepts remain the same.

    Uses group_id column if available to identify minimal pairs;
    falls back to across-class pairing within a concept dim.
    """
    print("[Test 3] Computing activations for modularity analysis ...")
    sae.eval()
    all_acts = []
    is_integrated = (sae_mode == "integrated")
    
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
            if is_integrated:
                z_batch = z_batch.flatten(1)
            a = sae.encode(z_batch)
            all_acts.append(a.cpu().numpy())
    A = np.vstack(all_acts)

    results = {}

    # Build reverse map: neuron -> list of concept keys
    neuron_to_concepts: Dict[int, List[str]] = defaultdict(list)
    for ck, nidx in assignment.items():
        neuron_to_concepts[nidx].append(ck)

    for dim in CONCEPT_DIMS:
        labels_arr = concept_labels[dim]
        n_values = len(value_names[dim])

        # Expected-off neurons for each value in this dim.
        dim_neurons = {
            vname: assignment[f"{dim}:{vname}"] for vname in value_names[dim]
        }
        # Neurons for OTHER dims (should stay same).
        other_dim_neurons = [
            assignment[f"{d}:{vname}"]
            for d in CONCEPT_DIMS if d != dim
            for vname in value_names[d]
        ]

        # Pair every sample with every other sample that differs ONLY in `dim`.
        # For efficiency, compare group means per (concept_val_in_dim, other_dims).
        for v_from_idx, v_from in enumerate(value_names[dim]):
            for v_to_idx, v_to in enumerate(value_names[dim]):
                if v_from_idx == v_to_idx:
                    continue
                mask_from = labels_arr == v_from_idx
                mask_to   = labels_arr == v_to_idx
                if not mask_from.any() or not mask_to.any():
                    continue

                A_from = A[mask_from]
                A_to   = A[mask_to]
                mean_from = A_from.mean(axis=0)
                mean_to   = A_to.mean(axis=0)
                delta = mean_to - mean_from

                n_off  = dim_neurons[v_from]
                n_on   = dim_neurons[v_to]

                delta_off   = float(delta[n_off])    # should be negative (turns off)
                delta_on    = float(delta[n_on])     # should be positive (turns on)
                delta_other = float(np.mean(np.abs(delta[other_dim_neurons])))  # should be ~0

                key = f"{dim}:{v_from}->{v_to}"
                results[key] = {
                    "delta_off":   delta_off,
                    "delta_on":    delta_on,
                    "delta_other_abs_mean": delta_other,
                }
                print(
                    f"  {key:40s} | off: {delta_off:+.4f} | on: {delta_on:+.4f} | "
                    f"other (abs): {delta_other:.4f}"
                )

    # Random neuron assignment baseline.
    # Precompute deltas between concept-value group means (reusable across random samples).
    concept_deltas = {}
    for dim in CONCEPT_DIMS:
        labels_arr = concept_labels[dim]
        for v_from_idx, v_from in enumerate(value_names[dim]):
            for v_to_idx, v_to in enumerate(value_names[dim]):
                if v_from_idx == v_to_idx:
                    continue
                mask_from = labels_arr == v_from_idx
                mask_to = labels_arr == v_to_idx
                if mask_from.any() and mask_to.any():
                    concept_deltas[(dim, v_from, v_to)] = A[mask_to].mean(0) - A[mask_from].mean(0)

    n_assigned = len(assignment)
    all_neurons = A.shape[1]
    rand_on, rand_off, rand_other = [], [], []

    for _ in range(N_BASELINE_RANDOM_SAMPLES):
        rand_neurons = np.random.choice(all_neurons, size=n_assigned, replace=False)
        rand_assign = {ck: int(rand_neurons[i]) for i, ck in enumerate(assignment.keys())}

        for dim in CONCEPT_DIMS:
            rd_neurons = {vn: rand_assign[f"{dim}:{vn}"] for vn in value_names[dim]}
            rd_other = [rand_assign[f"{d}:{vn}"] for d in CONCEPT_DIMS if d != dim for vn in value_names[d]]
            for v_from in value_names[dim]:
                for v_to in value_names[dim]:
                    if v_from == v_to:
                        continue
                    key = (dim, v_from, v_to)
                    if key not in concept_deltas:
                        continue
                    delta = concept_deltas[key]
                    rand_off.append(delta[rd_neurons[v_from]])
                    rand_on.append(delta[rd_neurons[v_to]])
                    rand_other.append(np.mean(np.abs(delta[rd_other])))

    results["baseline_random_delta_on"] = float(np.mean(rand_on))
    results["baseline_random_delta_off"] = float(np.mean(rand_off))
    results["baseline_random_delta_other_abs"] = float(np.mean(rand_other))
    print(
        f"  [Test 3 Baseline] Random neurons: on={results['baseline_random_delta_on']:+.4f}, "
        f"off={results['baseline_random_delta_off']:+.4f}, other={results['baseline_random_delta_other_abs']:.4f}"
    )

    return results


# ---------------------------------------------------------------------------
# 4. Test 4 — is_causal
# ---------------------------------------------------------------------------

def test_is_causal(
    sae,
    backbone: Optional[nn.Module],
    sae_layer: str,
    latents: th.Tensor,
    concept_labels: Dict[str, np.ndarray],
    value_names: Dict[str, List[str]],
    assignment: Dict[str, int],
    class_labels: th.Tensor,
    datadesc: pd.DataFrame,
    is_test: np.ndarray,
    sae_mode: str,
    device: str,
    batch_size: int = 512,
) -> Dict:
    """Test 4: is_causal — surgical ablation of concept neurons.

    For each concept dimension, zero out its assigned neurons in the SAE
    activations, reconstruct z_hat, and measure:
      (a) Accuracy on the ablated concept (should -> random).
      (b) Accuracy on all other concept dimensions (should remain high).

    backbone : the original CNN model (post-hoc: frozen;
               integrated: jointly trained). If None, we skip class-level eval.
    """
    if backbone is None:
        print("[Test 4] No backbone available; skipping.")
        return {}

    # Ensure all predictions correspond to the newly flattened A
    results = {}
    
    test_mask = is_test
    latents_test = latents[test_mask]
    class_labels_test = class_labels[test_mask]
    concept_labels_test = {k: v[test_mask] for k, v in concept_labels.items()}

    all_a = []
    is_integrated = (sae_mode == "integrated")
    
    with th.no_grad():
        for i in range(0, len(latents_test), batch_size):
            z_batch = latents_test[i : i + batch_size].float().to(device)
            if is_integrated:
                z_batch = z_batch.flatten(1)
            a = sae.encode(z_batch)
            all_a.append(a)
    A = th.cat(all_a, dim=0)  # (N_test, num_features)

    results = {}
    
    # T4 uses original latents when ablating
    Z_test = latents_test.clone().to(device)

    # Precompute class-to-concept maps (reused for real + baseline).
    all_classnames = sorted(datadesc.classname.unique())
    class_to_concept_maps = {}
    for dim in CONCEPT_DIMS:
        spec = specifications.get_specification_category(dim, all_classnames)
        mapping = np.zeros(len(all_classnames), dtype=int)
        for cn, s in spec.items():
            mapping[all_classnames.index(cn)] = s["classlabel"]
        class_to_concept_maps[dim] = mapping

    def _ablate_and_predict(A_in, neurons_to_zero):
        """Zero specified neurons, decode, forward through backbone → class preds."""
        A_mod = A_in.clone()
        A_mod[:, neurons_to_zero] = 0.0
        with th.no_grad():
            z_chunks = []
            for i in range(0, len(A_mod), batch_size):
                z_chunks.append(sae.decode(A_mod[i:i+batch_size].float().to(device)).cpu())
            Z_hat = th.cat(z_chunks, dim=0)
            l_chunks = []
            for i in range(0, len(Z_hat), batch_size):
                l_chunks.append(backbone.forward_from_layer(
                    Z_hat[i:i+batch_size].float().to(device), sae_layer).cpu())
        return th.cat(l_chunks, dim=0).argmax(dim=-1).numpy()

    # Real concept-neuron ablation.
    for ablate_dim in CONCEPT_DIMS:
        ablate_neurons = [
            assignment[f"{ablate_dim}:{vname}"] for vname in value_names[ablate_dim]
        ]
        preds = _ablate_and_predict(A_test, ablate_neurons)

        for eval_dim in CONCEPT_DIMS:
            eval_preds = class_to_concept_maps[eval_dim][preds]
            eval_labels = concept_labels_test[eval_dim]
            acc = (eval_preds == eval_labels).mean()
            random_baseline = 1.0 / len(value_names[eval_dim])
            key = f"ablate={ablate_dim}_eval={eval_dim}"
            results[key] = {
                "accuracy": float(acc),
                "random_baseline": random_baseline,
                "is_ablated_dim": eval_dim == ablate_dim,
            }
            tag = "<-- SHOULD BE RANDOM" if eval_dim == ablate_dim else ""
            print(
                f"  ablate={ablate_dim} | eval={eval_dim} | "
                f"acc={acc:.4f} (baseline={random_baseline:.3f}) {tag}"
            )

    # Random neuron ablation baseline.
    print(f"[Test 4 Baseline] Random ablation ({N_BASELINE_CAUSAL_SAMPLES} samples) ...")
    num_features = A_test.shape[1]
    for ablate_dim in CONCEPT_DIMS:
        n_to_ablate = len(value_names[ablate_dim])
        rand_ablated_accs = []
        rand_others_accs = []
        for _ in range(N_BASELINE_CAUSAL_SAMPLES):
            rand_neurons = np.random.choice(num_features, size=n_to_ablate, replace=False).tolist()
            rand_preds = _ablate_and_predict(A_test, rand_neurons)
            for eval_dim in CONCEPT_DIMS:
                rand_eval = class_to_concept_maps[eval_dim][rand_preds]
                rand_acc = float((rand_eval == concept_labels_test[eval_dim]).mean())
                if eval_dim == ablate_dim:
                    rand_ablated_accs.append(rand_acc)
                else:
                    rand_others_accs.append(rand_acc)
        results[f"baseline_random_ablate={ablate_dim}_ablated_acc"] = float(np.mean(rand_ablated_accs))
        results[f"baseline_random_ablate={ablate_dim}_others_acc"] = float(np.mean(rand_others_accs))
        print(
            f"  baseline ablate={ablate_dim} | ablated_acc={np.mean(rand_ablated_accs):.4f} | "
            f"others_acc={np.mean(rand_others_accs):.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# 5. Test 5 — Layer Probes (Integrated Models)
# ---------------------------------------------------------------------------

def _get_activations_at_layer(backbone, sae_layer, probe_layer, Z_hat, batch_size, device):
    """Forward Z_hat from sae_layer through the backbone and extract activations at probe_layer.

    Uses the backbone's own ``forward_from_layer`` / ``forward_to_layer`` helpers when
    possible, but falls back to manual encoder slicing so that we can stop at an
    *intermediate* layer rather than running all the way to logits.

    Works for any backbone that exposes ``get_cnn_activations``-style layer names and
    a Sequential ``encoder`` attribute (i.e. the CNN and ResNet models in this repo).
    """
    from dataset_transform import _format_layer

    # Build a mapping from layer name → encoder slice endpoint.
    # This is the same mapping used by dataset_transform.get_cnn_activations and
    # models.cnn.CNN.forward_to_layer, but we derive it dynamically so that it
    # adapts to any backbone with a Sequential encoder.
    all_layers = list(getattr(backbone, "_layer_slices", {}).keys())
    if not all_layers:
        # Fallback: reconstruct from the convention used by CNN / ResNet models.
        # Layer boundaries are recorded in forward_to_layer; mirror that logic.
        if hasattr(backbone, "forward_to_layer"):
            # CNN stores LAYER_SHAPES inside forward_from_layer; we just need
            # the slice *end* indices to know where each layer finishes.
            _slices = {
                "conv_layer0": (0, 2),
                "layer1":      (0, 5),
                "layer2":      (0, 8),
                "layer3":      (0, 11),
                "fc":          (0, len(backbone.encoder)),
            }
        else:
            raise RuntimeError("Cannot determine layer slices for this backbone.")
    else:
        _slices = backbone._layer_slices

    LAYER_SHAPES = {
        "conv_layer0": (64, 111, 111),
        "layer1":      (32, 55, 55),
        "layer2":      (16, 27, 27),
        "layer3":      (8,  13, 13),
    }

    def _unflatten_spatial(x, C, H, W):
        if x.ndim == 2:
            if x.shape[1] == C * H * W:
                x = x.unflatten(1, (C, H, W))
            elif x.shape[1] == H * W:
                x = x.unflatten(1, (H, W)).unsqueeze(1).expand(-1, C, -1, -1)
                x = x.contiguous()
        return x

    # Determine encoder index range: from end-of-sae_layer to end-of-probe_layer
    sae_end = _slices[sae_layer][1]
    probe_end = _slices[probe_layer][1]

    layer_acts = []
    with th.no_grad():
        for i in range(0, len(Z_hat), batch_size):
            z_hat_batch = Z_hat[i:i + batch_size].float().to(device)

            # Unflatten spatial layers so conv operations work
            if sae_layer in LAYER_SHAPES:
                C, H, W = LAYER_SHAPES[sae_layer]
                z_hat_batch = _unflatten_spatial(z_hat_batch, C, H, W)

            # Forward through the encoder slice [sae_end, probe_end)
            act = backbone.encoder[sae_end:probe_end](z_hat_batch)
            act_flat = _format_layer(act)
            layer_acts.append(act_flat.cpu())

    return th.cat(layer_acts, dim=0)


def test_downstream_probes(
    sae,
    backbone: nn.Module,
    sae_layer: str,
    latents: th.Tensor,
    is_test: np.ndarray,
    device: str,
    FLAGS,
    batch_size: int = 512,
) -> Dict:
    """Test 5: Layer Probes for Integrated Models.

    Evaluates linear probes on ALL layers of the backbone (not just downstream)
    when activations are generated by passing SAE-reconstructed features through
    the network.  This reveals whether the integrated sparse bottleneck affects
    concept representations at *every* layer — upstream, same, and downstream.

    For each probe layer we:
      1. Forward Z_hat (SAE reconstruction of the sae_layer activations) through
         the backbone from sae_layer → probe_layer.
      2. Load a pre-trained linear probe for that layer.
      3. Measure classification accuracy.

    Layers *upstream* of the SAE layer cannot be reached by forwarding Z_hat
    through the backbone (they precede the bottleneck).  For those layers we
    instead load the raw backbone activations directly — the interesting question
    is whether end-to-end integrated training changed those representations
    compared to a model without the bottleneck.
    """
    print("[Test 5] Probing ALL layers with integrated SAE...")
    sae.eval()
    backbone.eval()
    test_mask = is_test
    latents_test = latents[test_mask]

    # 1. Get Z_hat from the SAE
    all_zhat = []
    with th.no_grad():
        for i in range(0, len(latents_test), batch_size):
            z_batch = latents_test[i : i + batch_size].float().to(device)
            z_hat = sae(z_batch)
            all_zhat.append(z_hat.cpu())
    Z_hat = th.cat(all_zhat, dim=0)

    # 2. Determine all probe layers (exclude conv_layer0 — OOM on cluster)
    all_layers = flags.get_layers(FLAGS)
    probe_layers = [l for l in all_layers if l != "conv_layer0"]

    try:
        sae_layer_idx = all_layers.index(sae_layer)
    except ValueError:
        print(f"  sae_layer '{sae_layer}' not found in model layers; skipping.")
        return {}

    results = {}

    # Get labels for the test set
    FLAGS.dataspec = "classes"
    datadesc, ds_info = helpers.load_info(FLAGS)
    class_labels_test = th.tensor(datadesc.classlabel.values, dtype=th.long)[test_mask]

    from models import linear

    for probe_layer in probe_layers:
        probe_layer_idx = all_layers.index(probe_layer)

        if probe_layer_idx > sae_layer_idx:
            # --- Downstream of SAE: forward Z_hat through backbone ---
            print(f"  Probing layer (downstream): {probe_layer}")
            layer_acts = _get_activations_at_layer(
                backbone, sae_layer, probe_layer, Z_hat, batch_size, device
            )
        elif probe_layer_idx == sae_layer_idx:
            # --- Same layer as SAE: use Z_hat directly ---
            print(f"  Probing layer (sae layer):  {probe_layer}")
            layer_acts = Z_hat
        else:
            # --- Upstream of SAE: use raw backbone activations ---
            # These are the *integrated* backbone's own activations; comparing
            # them to a non-bottlenecked baseline shows whether integrated
            # training changed upstream representations.
            print(f"  Probing layer (upstream):   {probe_layer}")
            try:
                raw_path = flags.transformed_id(FLAGS, is_raw=True)
                layer_acts = helpers._get_vector_layer(raw_path, device, probe_layer)
                layer_acts = layer_acts[test_mask]
            except Exception as e:
                print(f"    Could not load upstream activations for {probe_layer}: {e}")
                continue

        # Load pre-trained linear probe for this layer
        try:
            train_id = flags.train_layerwise_id(FLAGS, probe_layer)
            probe_path = f"checkpoints/{train_id}.ckpt"
            if not os.path.exists(probe_path):
                print(f"    Probe not found: {probe_path}")
                continue

            probe = linear.Linear.load_from_checkpoint(probe_path)
            probe.eval()

            with th.no_grad():
                logits = probe(layer_acts)
                preds = logits.argmax(dim=-1)
                acc = (preds == class_labels_test).float().mean()
            
            results[probe_layer] = float(acc)
            print(f"    {probe_layer} accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"    Error evaluating probe for {probe_layer}: {e}")

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(FLAGS):
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load SAE checkpoint.
    sid = sae_id(FLAGS)
    ckpt_path = f"sae_checkpoints/{sid}.pt"
    print(f"[eval] Loading SAE checkpoint: {ckpt_path}")
    sae, head, payload = load_checkpoint(ckpt_path, device)
    sae.eval()
    if head is not None:
        head.eval()

    sae_layer = payload["sae_layer"]
    sae_mode = payload["sae_mode"]
    
    # Ensure FLAGS matches the loaded payload universally
    FLAGS.sae_mode = sae_mode

    # Load backbone for evaluation.
    if head is not None:
        # Integrated mode: backbone was trained jointly with SAE, saved in checkpoint.
        backbone = head
        print(f"[eval] Using integrated backbone from SAE checkpoint")
    else:
        # Post-hoc mode: load the separately pretrained backbone.
        from dataset_transform import load_raw_cnn
        backbone, _ = load_raw_cnn(FLAGS, device, FLAGS.data_path)
        print(f"[eval] Loaded pretrained backbone")
    backbone.eval()


    # Load latents and concept labels for main dataset.
    FLAGS.dataspec = "classes"
    (
        latents, class_labels, concept_labels, value_names,
        datadesc, dataspec, is_test, is_unseen
    ) = _load_all_latents_and_labels(FLAGS, sae_layer, device)

    # ------------------------------------------------------------------
    # Prerequisite: Concept Assignment
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PREREQUISITE: Concept Assignment (F1 bipartite graph)")
    print("="*60)
    assignment, f1_scores, graph = compute_concept_assignment(
        sae, latents, concept_labels, value_names, sae_mode=FLAGS.sae_mode, device=device
    )

    graph_path = f"results_sae/{FLAGS.jobname}/{sid}_concept_graph.json"
    os.makedirs(f"results_sae/{FLAGS.jobname}", exist_ok=True)
    with open(graph_path, "w") as fp:
        json.dump(graph, fp, indent=2)
    print(f"Saved concept assignment graph to {graph_path}")

    all_results = {
        "sae_id": sid,
        "concept_assignment": {ck: {"neuron": assignment[ck], "f1_seen": f1_scores[ck]}
                                for ck in assignment},
    }

    # ------------------------------------------------------------------
    # Test 1: is_grounded (uses T1 counterfactual minimal pairs dataset)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1: is_grounded")
    print("="*60)
    try:
        # T1 dataset has a different data_path ('t1' or 't1colors').
        data_path_train = FLAGS.data_path
        FLAGS.data_path = "t1colors" if "color" in data_path_train else "t1"
        FLAGS.dataspec = "classes"
        dataset_path_t1 = flags.transformed_id(
            FLAGS, 
            model_data_path="default" if FLAGS.data_path != "default" else None, 
            is_raw=True
        )
        latents_t1 = helpers._get_vector_layer(dataset_path_t1, device, sae_layer)
        if getattr(sae, "mode", None) == "integrated" or FLAGS.sae_mode == "integrated":
            latents_t1 = latents_t1.flatten(1)
        datadesc_t1, _ = helpers.load_info(FLAGS)
        class_labels_t1 = th.tensor(datadesc_t1.classlabel.values, dtype=th.long, device=device)
        
        FLAGS.data_path = data_path_train

        t1_results = test_is_grounded(
            sae, backbone, sae_layer, latents_t1, datadesc_t1,
            assignment, class_labels_t1, device
        )
        all_results["test1_is_grounded"] = t1_results
    except Exception as e:
        print(f"  [Test 1] Skipped: {e}")
        all_results["test1_is_grounded"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Test 2: is_token_of_type
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: is_token_of_type")
    print("="*60)
    t2_results = test_is_token_of_type(
        sae, latents, concept_labels, value_names, 
        assignment, is_unseen, sae_mode=FLAGS.sae_mode, device=device
    )
    all_results["is_token_of_type"] = t2_results

    # ------------------------------------------------------------------
    # Test 3: is_modular
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3: is_modular")
    print("="*60)
    t3_results = test_is_modular(
        sae, latents, concept_labels, value_names, 
        assignment, datadesc, sae_mode=FLAGS.sae_mode, device=device
    )
    all_results["modularity"] = t3_results

    # ------------------------------------------------------------------
    # Test 4: is_causal
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 4: is_causal")
    print("="*60)
    t4_results = test_is_causal(
        sae, backbone, sae_layer, latents, concept_labels, value_names, 
        assignment, class_labels, datadesc, is_test, sae_mode=FLAGS.sae_mode, device=device
    )
    all_results["is_causal"] = t4_results

    # ------------------------------------------------------------------
    # Test 5: Layer Probes (Integrated Models — all layers)
    # ------------------------------------------------------------------
    if sae_mode == "integrated" and backbone is not None:
        print("\n" + "="*60)
        print("TEST 5: Layer Probes (Integrated — all layers)")
        print("="*60)
        t5_results = test_downstream_probes(
            sae, backbone, sae_layer, latents, is_test, device, FLAGS
        )
        all_results["test5_downstream_probes"] = t5_results
    else:
        print("\n" + "="*60)
        print("TEST 5: Layer Probes (Skipped — post-hoc mode)")
        print("="*60)

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    out_path = f"results_sae/{FLAGS.jobname}/{sid}_eval.json"
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import flags
    parser = get_sae_flags()
    FLAGS = parser.parse_args()
    main(FLAGS)
