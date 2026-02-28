#!/bin/bash
#SBATCH -J sae_unit_tests
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./err/%x-%j.out

echo "Loading python module and conda..."
module load python
echo "Setting up Conda Environment..."
eval "$(conda shell.bash hook)"

ENV_NAME="sparse_bottlenecks"
echo "Activating ${ENV_NAME} environment..."
conda activate /n/ba_lab/Everyone/cmcnamee/conda/envs/${ENV_NAME} || conda activate ${ENV_NAME} || echo "Failed to activate ${ENV_NAME}!"

pip install scikit-learn

# IMPORTANT: Run this from the code/gaila directory
cd /n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila || echo "Please run from unit_tests/code/gaila"

DATE="sae_experiments"
mkdir -p results_sae/${DATE}
mkdir -p sae_checkpoints

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The pretrained CNN backbone must already exist from run_from_scratch.sh.
# Layers of the RawCnn backbone available for SAE attachment:
#   conv_layer0, layer1, layer2, layer3, fc
# ---------------------------------------------------------------------------

SEEDS="6 7 8 9 10"
LAYERS="conv_layer0 layer1 layer2 layer3 fc"
SAE_ARCHS="standard topk"    # add 'gated' to test that too
SAE_MODES="post_hoc integrated"
SAE_FEATURES=4096             # 8x the 512-dim latent space
L1_COEFF=1e-3
TOPK_K=32
SAE_EPOCHS=50
SAE_LR=1e-4

# ---------------------------------------------------------------------------
# Train SAEs at every layer, for every seed, for every mode and architecture
# ---------------------------------------------------------------------------
echo "=== Training SAEs ==="

for SEED in $SEEDS; do
    for LAYER in $LAYERS; do
        for MODE in $SAE_MODES; do
            for ARCH in $SAE_ARCHS; do
                echo "  [SAE TRAIN] mode=$MODE arch=$ARCH layer=$LAYER seed=$SEED"
                python sae_train.py \
                    --pretrain_model RawCnn \
                    --finetune_model linear \
                    --data_path default \
                    --seed $SEED \
                    --n_epochs $SAE_EPOCHS \
                    --samples_per_class 1000 \
                    --jobname $DATE \
                    --batch_size 256 \
                    --sae_mode $MODE \
                    --sae_arch $ARCH \
                    --sae_layer $LAYER \
                    --sae_num_features $SAE_FEATURES \
                    --sae_l1_coeff $L1_COEFF \
                    --sae_topk_k $TOPK_K \
                    --sae_lr $SAE_LR
            done
        done
    done
done

# ---------------------------------------------------------------------------
# Evaluate SAEs: Prerequisite + Tests 1-4 for every trained SAE
# ---------------------------------------------------------------------------
echo "=== Evaluating SAEs ==="

for SEED in $SEEDS; do
    for LAYER in $LAYERS; do
        for MODE in $SAE_MODES; do
            for ARCH in $SAE_ARCHS; do
                # Use 1-slice holdout (matches original paper's 1-slice setting)
                # for the is_token_of_type test OOD evaluation.
                echo "  [SAE EVAL] mode=$MODE arch=$ARCH layer=$LAYER seed=$SEED"
                python sae_eval.py \
                    --pretrain_model RawCnn \
                    --finetune_model linear \
                    --data_path default \
                    --seed $SEED \
                    --n_epochs $SAE_EPOCHS \
                    --samples_per_class 1000 \
                    --jobname $DATE \
                    --batch_size 256 \
                    --sae_mode $MODE \
                    --sae_arch $ARCH \
                    --sae_layer $LAYER \
                    --sae_num_features $SAE_FEATURES \
                    --sae_l1_coeff $L1_COEFF \
                    --sae_topk_k $TOPK_K \
                    --sae_lr $SAE_LR \
                    --layout_inlp_or_holdout oval \
                    --shape_inlp_or_holdout clean
            done
        done
    done
done

echo "Done! Results are in results_sae/${DATE}/"
