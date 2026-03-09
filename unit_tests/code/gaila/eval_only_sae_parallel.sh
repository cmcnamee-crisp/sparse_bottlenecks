#!/bin/bash
# ---------------------------------------------------------------------------
# Parallel SAE Evaluation: one SLURM array task per (seed, layer, mode, arch)
#
# 80 combinations = 5 seeds × 4 layers × 2 modes × 2 archs
# Array index mapping (innermost varies fastest):
#   arch_idx  = TASK_ID % 2
#   mode_idx  = (TASK_ID / 2) % 2
#   layer_idx = (TASK_ID / 4) % 4
#   seed_idx  = (TASK_ID / 16) % 5
#
# All 80 jobs submitted at once (well within cluster norms per FASRC docs).
# Estimated ~15 min per combo; 30 min wall time gives 2× headroom.
# Memory reduced from 32G → 16G (peak usage ~12-15G for causal tests).
# Memory reduced from 32G → 16G (peak usage ~12-15G for causal tests).
# ---------------------------------------------------------------------------

set -eo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Will print commands without submitting."
fi

PROJECT_DIR="/n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila"
CONDA_ENV="/n/ba_lab/Everyone/cmcnamee/conda/envs/sparse_bottlenecks"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"

mkdir -p "${LOGS_DIR}"

echo "Loading python module and conda..."
module load python
echo "Setting up Conda Environment..."
eval "$(conda shell.bash hook)"

ENV_NAME="sparse_bottlenecks"
echo "Activating ${ENV_NAME} environment..."
conda activate /n/ba_lab/Everyone/cmcnamee/conda/envs/${ENV_NAME} || conda activate ${ENV_NAME} || echo "Failed to activate ${ENV_NAME}!"

# IMPORTANT: Run this from the code/gaila directory
cd /n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila || echo "Please run from unit_tests/code/gaila"

DATE="sae_experiments"
RUN_SAE_EVAL=false   # Set to true to run sae_eval.py (probe-free neuron matching)

# ---------------------------------------------------------------------------
# Configuration (Must match run_sae.sh)
# ---------------------------------------------------------------------------
SEEDS=(6 7 8 9 10)
LAYERS=(layer1 layer2 layer3 fc)
SAE_MODES=(post_hoc integrated)
SAE_ARCHS=(standard topk)
SAE_FEATURES=4096
L1_COEFF=1e-3
TOPK_K=32
PRETRAIN_EPOCHS=100
SAE_LR=1e-4

# ---------------------------------------------------------------------------

TRANSFORM_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH -J sae_data_transform
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./err/%x-%j.out

module load python
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV} || { echo "FATAL: could not activate env"; exit 1; }

cd ${PROJECT_DIR}

echo "=== Phase 1: Base Dataset Transforms ==="
for DATA_PATH in default t1 t2; do
    echo "  -> dataset_transform.py --data_path \$DATA_PATH"
    python dataset_transform.py --pretrain_model RawCnn --data_path \$DATA_PATH --seed 6 --n_epochs 100 --samples_per_class 1000 --layerwise True --jobname sae_experiments
done
echo "=== Phase 1 Complete ==="
EOF
)

EVAL_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH -J sae_eval_par
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-79
#SBATCH -o ./out/%x-%A_%a.out
#SBATCH -e ./err/%x-%A_%a.out

module load python
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV} || { echo "FATAL: could not activate env"; exit 1; }

cd ${PROJECT_DIR}

TASK_ID=\${SLURM_ARRAY_TASK_ID}

ARCH_IDX=\$(( TASK_ID % 2 ))
MODE_IDX=\$(( (TASK_ID / 2) % 2 ))
LAYER_IDX=\$(( (TASK_ID / 4) % 4 ))
SEED_IDX=\$(( (TASK_ID / 16) % 5 ))

SEEDS=(6 7 8 9 10)
LAYERS=(layer1 layer2 layer3 fc)
SAE_MODES=(post_hoc integrated)
SAE_ARCHS=(standard topk)

SEED=\${SEEDS[\$SEED_IDX]}
LAYER=\${LAYERS[\$LAYER_IDX]}
MODE=\${SAE_MODES[\$MODE_IDX]}
ARCH=\${SAE_ARCHS[\$ARCH_IDX]}

DATE="sae_experiments"
PRETRAIN_EPOCHS=100
SAE_FEATURES=4096
L1_COEFF=1e-3
TOPK_K=32
SAE_LR=1e-4

echo "=== Array task \${TASK_ID}: seed=\${SEED} layer=\${LAYER} mode=\${MODE} arch=\${ARCH} ==="

# ---------------------------------------------------------------------------
# Step 1: Dataset transform (default + t1 + t2) for SAE logic
# ---------------------------------------------------------------------------
echo "  [SAE DATASET TRANSFORM default] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python sae_dataset_transform.py --pretrain_model RawCnn --data_path default --model_data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --layerwise True --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

echo "  [SAE DATASET TRANSFORM t1] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python sae_dataset_transform.py --pretrain_model RawCnn --data_path t1 --model_data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --layerwise True --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

echo "  [SAE DATASET TRANSFORM t2] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python sae_dataset_transform.py --pretrain_model RawCnn --data_path t2 --model_data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --layerwise True --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

# ---------------------------------------------------------------------------
# Step 2: Train linear probes
# ---------------------------------------------------------------------------
echo "  [SAE PROBE TRAIN default] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python train.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

echo "  [SAE PROBE TRAIN t2] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python train.py --pretrain_model RawCnn --finetune_model linear --data_path t2 --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

# ---------------------------------------------------------------------------
# Step 3: Run T1 and T3/T4 tests
# ---------------------------------------------------------------------------
echo "  [SAE T1/T3/T4 TESTS] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
python sae_t1.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

python sae_t3t4.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --jobname \$DATE --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K

# ---------------------------------------------------------------------------
# Step 4 (optional): Probe-free neuron matching
# ---------------------------------------------------------------------------
if [ "\$RUN_SAE_EVAL" = true ]; then
    echo "  [SAE EVAL HOOK] mode=\$MODE arch=\$ARCH layer=\$LAYER seed=\$SEED"
    python sae_eval.py --pretrain_model RawCnn --data_path default --seed \$SEED --n_epochs \$PRETRAIN_EPOCHS --samples_per_class 1000 --jobname \$DATE --batch_size 256 --sae_mode \$MODE --sae_arch \$ARCH --sae_layer \$LAYER --sae_num_features \$SAE_FEATURES --sae_l1_coeff \$L1_COEFF --sae_topk_k \$TOPK_K --sae_lr \$SAE_LR --layout_inlp_or_holdout column --shape_inlp_or_holdout oval --stroke_inlp_or_holdout clean
fi

echo "=== Done: seed=\${SEED} layer=\${LAYER} mode=\${MODE} arch=\${ARCH} ==="
EOF
)

EVAL_SBATCH=$(mktemp /tmp/sae_eval_XXXXXX.sh)
TRANSFORM_SBATCH=$(mktemp /tmp/sae_data_transform_XXXXXX.sh)

echo "${EVAL_SCRIPT}" > "${EVAL_SBATCH}"
echo "${TRANSFORM_SCRIPT}" > "${TRANSFORM_SBATCH}"

if $DRY_RUN; then
    echo "=== Phase 1: Base Dataset Transform Job ==="
    echo "  Script: ${TRANSFORM_SBATCH}"
    echo "  (would run: sbatch ${TRANSFORM_SBATCH})"

    echo "=== Phase 2: Array Eval Job ==="
    echo "  Script: ${EVAL_SBATCH}"
    echo "  (would run: sbatch --dependency=afterok:DRYRUN ${EVAL_SBATCH})"
else
    echo "=== Phase 1: Submitting dataset transform job ==="
    TRANSFORM_JOB_ID=$(sbatch --parsable "${TRANSFORM_SBATCH}")
    echo "  Dataset transform job submitted: ${TRANSFORM_JOB_ID}"

    echo "=== Phase 2: Submitting array eval job ==="
    EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:${TRANSFORM_JOB_ID} "${EVAL_SBATCH}")
    echo "  Array eval job submitted: ${EVAL_JOB_ID}"
fi
