#!/bin/bash
# run_sae.sh — Parallel SAE training & evaluation via SLURM job arrays
#
# Spawns one SLURM job per (seed, layer, mode, arch) combination so that
# all SAE model trainings run in parallel on the cluster.
#
# Usage:
#   cd /n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila
#   bash run_sae.sh              # submit all jobs
#   bash run_sae.sh --dry-run    # print what would be submitted without submitting
#
# Prerequisites:
#   - Pretrained RawCnn checkpoints must already exist (from run_from_scratch.sh)
#   - Dataset transforms are run as a blocking prerequisite before the parallel jobs

set -eo pipefail

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Will print commands without submitting."
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="/n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila"
CONDA_ENV="/n/ba_lab/Everyone/cmcnamee/conda/envs/sparse_bottlenecks"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"

mkdir -p "${LOGS_DIR}"

# ---------------------------------------------------------------------------
# Configuration  (must match run_from_scratch.sh settings)
# ---------------------------------------------------------------------------
SEEDS=(6 7 8 9 10)
LAYERS=(layer1 layer2 layer3 fc)
SAE_ARCHS=(standard topk)
SAE_MODES=(post_hoc integrated)
SAE_FEATURES=4096
L1_COEFF=1e-3
TOPK_K=32
SAE_EPOCHS=100
SAE_LR=1e-4
PRETRAIN_EPOCHS=100

DATE="sae_experiments"

# ---------------------------------------------------------------------------
# Phase 0 — Ensure output dirs exist
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/results_sae/${DATE}"
mkdir -p "${PROJECT_DIR}/sae_checkpoints"

# ---------------------------------------------------------------------------
# Phase 1 — Dataset Transforms (sequential prerequisite)
# ---------------------------------------------------------------------------
# These are fast and must complete before any training job starts, so we run
# them as a single blocking SLURM job (or locally if you prefer).
# ---------------------------------------------------------------------------

TRANSFORM_SCRIPT=$(cat <<'HEREDOC'
#!/bin/bash
#SBATCH -J sae_transforms
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o __LOGS_DIR__/transforms-%j.out
#SBATCH -e __LOGS_DIR__/transforms-%j.err

module load python
eval "$(conda shell.bash hook)"
conda activate __CONDA_ENV__ || { echo "FATAL: could not activate env"; exit 1; }

cd __PROJECT_DIR__

__TRANSFORM_CMDS__

echo "=== Dataset transforms complete ==="
HEREDOC
)

# Build the transform commands for every seed
TRANSFORM_CMDS=""
for SEED in "${SEEDS[@]}"; do
    TRANSFORM_CMDS+="echo \"  [TRANSFORM] seed=${SEED} data_path=default\"
python dataset_transform.py \\
    --pretrain_model RawCnn \\
    --data_path default \\
    --seed ${SEED} \\
    --n_epochs ${PRETRAIN_EPOCHS} \\
    --samples_per_class 1000 \\
    --jobname ${DATE} \\
    --batch_size 256 \\
    --layerwise True || true

echo "  [TRANSFORM] seed=${SEED} data_path=t1"
python dataset_transform.py \
    --pretrain_model RawCnn \
    --data_path t1 \
    --model_data_path default \
    --seed ${SEED} \
    --n_epochs ${PRETRAIN_EPOCHS} \
    --samples_per_class 1000 \
    --jobname ${DATE} \
    --batch_size 256 \
    --layerwise True || true

echo "  [TRANSFORM] seed=${SEED} data_path=t2"
python dataset_transform.py \
    --pretrain_model RawCnn \
    --data_path t2 \
    --model_data_path default \
    --seed ${SEED} \
    --n_epochs ${PRETRAIN_EPOCHS} \
    --samples_per_class 1000 \
    --jobname ${DATE} \
    --batch_size 256 \
    --layerwise True || true

"
done

# Substitute placeholders into the transform script
TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT//__LOGS_DIR__/${LOGS_DIR}}"
TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT//__CONDA_ENV__/${CONDA_ENV}}"
TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT//__PROJECT_DIR__/${PROJECT_DIR}}"
TRANSFORM_SCRIPT="${TRANSFORM_SCRIPT//__TRANSFORM_CMDS__/${TRANSFORM_CMDS}}"

TRANSFORM_SBATCH=$(mktemp /tmp/sae_transform_XXXXXX.sh)
echo "${TRANSFORM_SCRIPT}" > "${TRANSFORM_SBATCH}"

if $DRY_RUN; then
    echo ""
    echo "=== Phase 1: Dataset Transform Job ==="
    echo "  Script: ${TRANSFORM_SBATCH}"
    echo "  (would run: sbatch ${TRANSFORM_SBATCH})"
    TRANSFORM_JOB_ID="DRYRUN"
else
    echo "=== Phase 1: Submitting dataset transform job ==="
    TRANSFORM_JOB_ID=$(sbatch --parsable "${TRANSFORM_SBATCH}")
    echo "  Transform job submitted: ${TRANSFORM_JOB_ID}"
fi

# ---------------------------------------------------------------------------
# Phase 2 — Parallel SAE Training Jobs
# ---------------------------------------------------------------------------
# One job per (seed, layer, mode, arch) combination.
# Each job depends on the transform job completing successfully.
# ---------------------------------------------------------------------------

echo ""
echo "=== Phase 2: Submitting SAE training jobs ==="

TRAIN_JOB_IDS=()
JOB_COUNT=0

for SEED in "${SEEDS[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for MODE in "${SAE_MODES[@]}"; do
            for ARCH in "${SAE_ARCHS[@]}"; do
                JOB_NAME="sae_${MODE}_${ARCH}_${LAYER}_s${SEED}"

                TRAIN_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o ${LOGS_DIR}/${JOB_NAME}-%j.out
#SBATCH -e ${LOGS_DIR}/${JOB_NAME}-%j.err

module load python
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV} || { echo "FATAL: could not activate env"; exit 1; }

pip install scikit-learn 2>/dev/null

cd ${PROJECT_DIR}

echo "=== SAE Training: mode=${MODE} arch=${ARCH} layer=${LAYER} seed=${SEED} ==="
python sae_train.py \\
    --pretrain_model RawCnn \\
    --data_path default \\
    --seed ${SEED} \\
    --n_epochs ${SAE_EPOCHS} \\
    --samples_per_class 1000 \\
    --jobname ${DATE} \\
    --batch_size 256 \\
    --sae_mode ${MODE} \\
    --sae_arch ${ARCH} \\
    --sae_layer ${LAYER} \\
    --sae_num_features ${SAE_FEATURES} \\
    --sae_l1_coeff ${L1_COEFF} \\
    --sae_topk_k ${TOPK_K} \\
    --sae_lr ${SAE_LR}

echo "=== SAE Training complete, starting evaluation ==="
python sae_eval.py \\
    --pretrain_model RawCnn \\
    --data_path default \\
    --seed ${SEED} \\
    --n_epochs ${PRETRAIN_EPOCHS} \\
    --samples_per_class 1000 \\
    --jobname ${DATE} \\
    --batch_size 256 \\
    --sae_mode ${MODE} \\
    --sae_arch ${ARCH} \\
    --sae_layer ${LAYER} \\
    --sae_num_features ${SAE_FEATURES} \\
    --sae_l1_coeff ${L1_COEFF} \\
    --sae_topk_k ${TOPK_K} \\
    --sae_lr ${SAE_LR} \\
    --layout_inlp_or_holdout column \\
    --shape_inlp_or_holdout oval \\
    --stroke_inlp_or_holdout clean

echo "=== Done: ${JOB_NAME} ==="
EOF
)

                TRAIN_SBATCH=$(mktemp /tmp/sae_train_XXXXXX.sh)
                echo "${TRAIN_SCRIPT}" > "${TRAIN_SBATCH}"

                if $DRY_RUN; then
                    echo "  [${JOB_COUNT}] ${JOB_NAME}  (would submit with --dependency=afterok:DRYRUN)"
                else
                    JID=$(sbatch --parsable --dependency=afterok:${TRANSFORM_JOB_ID} "${TRAIN_SBATCH}")
                    TRAIN_JOB_IDS+=("${JID}")
                    echo "  [${JOB_COUNT}] ${JOB_NAME} -> job ${JID}  (depends on ${TRANSFORM_JOB_ID})"
                fi

                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
done

echo ""
echo "=== Summary ==="
echo "  Transform job:  ${TRANSFORM_JOB_ID:-N/A}"
echo "  Training jobs:  ${JOB_COUNT} submitted"
echo "  Logs:           ${LOGS_DIR}/"
echo "  Results:        results_sae/${DATE}/"
echo ""

if ! $DRY_RUN; then
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  sacct -j ${TRANSFORM_JOB_ID} --format=JobID,JobName,State,Elapsed"
fi

echo "Done!"
