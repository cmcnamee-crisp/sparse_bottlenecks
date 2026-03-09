#!/bin/bash
#SBATCH -J gen_t2
#SBATCH -p seas_gpu
#SBATCH --account=ba_lab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./err/%x-%j.out

echo "Loading python module and conda..."
module load python
echo "Setting up Conda Environment..."
eval "$(conda shell.bash hook)"

ENV_NAME="sparse_bottlenecks"
echo "Activating sparse_bottlenecks environment..."
conda activate /n/ba_lab/Everyone/cmcnamee/conda/envs/${ENV_NAME} || conda activate ${ENV_NAME} || echo "Failed to activate ${ENV_NAME}!"

# IMPORTANT: Run this from the code/gaila directory
cd /n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila || echo "Please run from unit_tests/code/gaila"

# 2. Generate Dataset for "t2"
if [ ! -f "data/datadesc_t2_6.tsv" ]; then
    echo "Generating 't2' synthetic dataset..."
    for SEED in 6 7 8 9 10; do
        python dataset.py --data_path t2 --seed $SEED
    done
else
    echo "'t2' synthetic dataset already exists! Skipping generation..."
fi

echo "Done! Run eval_only_sae_parallel afterwards."
