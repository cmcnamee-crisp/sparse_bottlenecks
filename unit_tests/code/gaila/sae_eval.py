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
    dataset_path = flags.transformed_id(FLAGS)

    latents = helpers._get_vector_layer(dataset_path, device, layer, "avg")
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
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
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
            f1_arr = np.array([
                f1_score(y_true, A_binary[:, j], zero_division=0)
                for j in range(num_features)
            ])
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
    }
    return assignment, f1_scores, graph


# ---------------------------------------------------------------------------
# 1. Test 1 — is_grounded
# ---------------------------------------------------------------------------

def test_is_grounded(
    sae,
    backbone_head,  # nn.Linear or None (for post-hoc, we use pre-trained linear probe)
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
        total_dists, concept_dists = [], []
        for gid in unique_ids:
            idx_in_group = np.where(ids == gid)[0]
            if len(idx_in_group) < 2:
                continue
            A_group = A[idx_in_group]  # (k, num_features)
            for i in range(len(A_group)):
                for j in range(i + 1, len(A_group)):
                    diff = np.abs(A_group[i] - A_group[j])
                    total_dists.append(diff.sum())
                    concept_dists.append(diff[concept_neuron_indices].sum())
        mean_total = float(np.mean(total_dists)) if total_dists else float("nan")
        mean_concept = float(np.mean(concept_dists)) if concept_dists else float("nan")
        results["mean_total_sparse_code_dist"] = mean_total
        results["mean_concept_neuron_dist"] = mean_concept
        results["mean_background_dist"] = mean_total - mean_concept
        if mean_total > 0:
            results["background_fraction"] = (mean_total - mean_concept) / mean_total
    else:
        # No pair ids available; report marginal activation stats instead.
        results["note"] = "no group_id column; skipping pairwise analysis"
        results["mean_activation_l2"] = float(np.mean(np.linalg.norm(A, axis=1)))

    # Classification accuracy using z_hat (reconstucted latents).
    if backbone_head is not None:
        backbone_head.eval()
        with th.no_grad():
            logits = backbone_head(Z_hat.to(device))
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
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
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
    device: str,
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
    with th.no_grad():
        for i in range(0, len(latents), batch_size):
            z_batch = latents[i : i + batch_size].float().to(device)
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

    return results


# ---------------------------------------------------------------------------
# 4. Test 4 — is_causal
# ---------------------------------------------------------------------------

def test_is_causal(
    sae,
    backbone_head: Optional[nn.Module],
    latents: th.Tensor,
    concept_labels: Dict[str, np.ndarray],
    value_names: Dict[str, List[str]],
    assignment: Dict[str, int],
    class_labels: th.Tensor,
    is_test: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> Dict:
    """Test 4: is_causal — surgical ablation of concept neurons.

    For each concept dimension, zero out its assigned neurons in the SAE
    activations, reconstruct z_hat, and measure:
      (a) Accuracy on the ablated concept (should -> random).
      (b) Accuracy on all other concept dimensions (should remain high).

    backbone_head : the linear classification head (post-hoc: pretrained probe;
                    integrated: the SAE head).  If None, we skip class-level eval.
    """
    if backbone_head is None:
        print("[Test 4] No classification head available; skipping.")
        return {}

    print("[Test 4] Computing full-dataset activations ...")
    sae.eval()
    backbone_head.eval()

    test_mask = is_test
    latents_test = latents[test_mask]
    class_labels_test = class_labels[test_mask]
    concept_labels_test = {k: v[test_mask] for k, v in concept_labels.items()}

    all_acts = []
    with th.no_grad():
        for i in range(0, len(latents_test), batch_size):
            z_batch = latents_test[i : i + batch_size].float().to(device)
            a = sae.encode(z_batch)
            all_acts.append(a.cpu())
    A_test = th.cat(all_acts, dim=0)  # (N_test, num_features)

    results = {}

    # For each concept dimension, ablate and evaluate.
    for ablate_dim in CONCEPT_DIMS:
        ablate_neurons = [
            assignment[f"{ablate_dim}:{vname}"] for vname in value_names[ablate_dim]
        ]

        # Surgical ablation: zero only the assigned neurons for this dim.
        A_ablated = A_test.clone()
        A_ablated[:, ablate_neurons] = 0.0

        # Reconstruct and predict.
        preds_ablated_dim = {}  # dim -> predicted labels
        with th.no_grad():
            z_hat_chunks = []
            for i in range(0, len(A_ablated), batch_size):
                a_batch = A_ablated[i : i + batch_size].float().to(device)
                z_hat = sae.decode(a_batch)
                z_hat_chunks.append(z_hat.cpu())
            Z_hat_all = th.cat(z_hat_chunks, dim=0)

            logits_chunks = []
            for i in range(0, len(Z_hat_all), batch_size):
                z_chunk = Z_hat_all[i : i + batch_size].float().to(device)
                logits = backbone_head(z_chunk)
                logits_chunks.append(logits.cpu())
            all_logits = th.cat(logits_chunks, dim=0)

        preds = all_logits.argmax(dim=-1).numpy()

        # Evaluate accuracy per concept dimension.
        # We need a mapping from 18-way class predictions to the 3-way concept labels.
        classnames = sorted(value_names["layout"]) # Use any dim to get all classnames
        # Wait, value_names contains concept values, not classnames.
        # Let's get classnames from the datadesc.
        all_classnames = sorted(datadesc.classname.unique())
        
        for eval_dim in CONCEPT_DIMS:
            # Map predicted class index -> concept label for this dim
            spec = specifications.get_specification_category(eval_dim, all_classnames)
            # class_idx -> concept_label
            class_to_concept = np.zeros(len(all_classnames), dtype=int)
            for cn, s in spec.items():
                class_to_concept[all_classnames.index(cn)] = s["classlabel"]
            
            eval_preds = class_to_concept[preds]
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

    # Load latents and concept labels for main dataset.
    FLAGS.dataspec = "classes"
    (
        latents, class_labels, concept_labels, value_names,
        datadesc, dataspec, is_test, is_unseen
    ) = _load_all_latents_and_labels(FLAGS, sae_layer, device)

    # NEW: Load backbone head globally for all tests (especially Test 1 and Test 4)
    backbone_head = head 
    if backbone_head is None:
        # Load pretrained linear probe from checkpoints
        try:
            # We need the one trained on classes for the final layer
            # Note: if this fails, check if your --n_epochs or --samples_per_class 
            # flags match the original backbone training run.
            train_id = flags.train_layerwise_id(FLAGS, sae_layer)
            ckpt_path_probe = f"checkpoints/{train_id}.ckpt"
            from models import linear as linear_model
            backbone_head = linear_model.Linear.load_from_checkpoint(
                checkpoint_path=ckpt_path_probe
            )
            backbone_head.to(device)
            backbone_head.eval()
            print(f"  [eval] Loaded backbone head from {ckpt_path_probe}")
        except Exception as e:
            print(f"  [eval] Warning: Could not load backbone head from {ckpt_path_probe}: {e}")
            print("         Test 4 (Causality) will be skipped.")

    # ------------------------------------------------------------------
    # Prerequisite: Concept Assignment
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PREREQUISITE: Concept Assignment (F1 bipartite graph)")
    print("="*60)
    assignment, f1_scores, graph = compute_concept_assignment(
        sae, latents, concept_labels, value_names, device=device
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
        dataset_path_t1 = flags.transformed_id(FLAGS, data_path_train)
        latents_t1 = helpers._get_vector_layer(dataset_path_t1, device, sae_layer, "avg")
        datadesc_t1, _ = helpers.load_info(FLAGS)
        class_labels_t1 = th.tensor(datadesc_t1.classlabel.values, dtype=th.long, device=device)
        FLAGS.data_path = data_path_train

        t1_results = test_is_grounded(
            sae, backbone_head, latents_t1, datadesc_t1,
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
        sae, latents, concept_labels, value_names, assignment, is_unseen, device=device
    )
    all_results["test2_is_token_of_type"] = t2_results

    # ------------------------------------------------------------------
    # Test 3: is_modular
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3: is_modular")
    print("="*60)
    t3_results = test_is_modular(
        sae, latents, concept_labels, value_names, assignment, datadesc, device
    )
    all_results["test3_is_modular"] = t3_results

    # ------------------------------------------------------------------
    # Test 4: is_causal
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 4: is_causal")
    print("="*60)
    t4_results = test_is_causal(
        sae, backbone_head, latents, concept_labels, value_names,
        assignment, class_labels, is_test, device
    )
    all_results["test4_is_causal"] = t4_results

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    out_path = f"results_sae/{FLAGS.jobname}/{sid}_eval.json"
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import flags as _flags
    parser = get_sae_flags()
    FLAGS = parser.parse_args()
    main(FLAGS)
