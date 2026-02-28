#!/bin/bash
#SBATCH -J true_from_scratch
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
echo "Activating sparse_bottlenecks environment..."
conda activate /n/ba_lab/Everyone/cmcnamee/conda/envs/${ENV_NAME} || conda activate ${ENV_NAME} || echo "Failed to activate ${ENV_NAME}!"

# Install required packages for this experiment into the existing environment
echo "Ensuring required packages are installed..."
pip install ftfy regex tqdm pytorch-lightning pandas seaborn pytest matplotlib wandb scipy scikit-learn
pip install git+https://github.com/openai/CLIP.git

# IMPORTANT: Run this from the code/gaila directory
cd /n/ba_lab/Everyone/cmcnamee/sparse_bottlenecks/unit_tests/code/gaila || echo "Please run from unit_tests/code/gaila"

# 1. Setup required directories
echo "Setting up output directories..."
bash setup.sh

# Download the pre-computed features (ds.zip) if we don't already have them
if [ ! -d "data-slimmer" ]; then
    echo "Downloading ds.zip to the cluster..."
    # The Google Drive direct download URL for the ds.zip file
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=132i_yI-Xf9u-vV1fL0C5hGqO6R3Lz9E0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=132i_yI-Xf9u-vV1fL0C5hGqO6R3Lz9E0" -O ds.zip && rm -rf /tmp/cookies.txt
    
    echo "Unzipping data..."
    unzip -q ds.zip
    rm ds.zip
fi

# 2. Generate Dataset for "default"
if [ ! -f "data/datadesc_default_6.tsv" ]; then
    echo "Generating 'default' synthetic dataset..."
    for SEED in 6 7 8 9 10; do
        python dataset.py --data_path default --seed $SEED
    done
else
    echo "'default' synthetic dataset already exists! Skipping generation..."
fi

if [ ! -f "data/datadesc_t1_6.tsv" ]; then
    echo "Generating 't1' synthetic dataset..."
    for SEED in 6 7 8 9 10; do
        python dataset.py --data_path t1 --seed $SEED
    done
else
    echo "'t1' synthetic dataset already exists! Skipping generation..."
fi

# 3. Train the From-Scratch CNN (RawCnn)
echo "Pretraining RawCnn..."
DATE="final_scratch"
# Train the CNN on the 5 seeds
for SEED in 6 7 8 9 10; do
    python pretrain.py --pretrain_model RawCnn --data_path default --seed $SEED --n_epochs 100 --dataspec classes --jobname $DATE --samples_per_class 1000
done

# 3.5. Transform Dataset
# This MUST happen after pretrain.py, because dataset_transform tries to encode images through the trained RawCnn
echo "Transforming dataset using trained models..."
for SEED in 6 7 8 9 10; do
    python dataset_transform.py --pretrain_model RawCnn --data_path default --seed $SEED --n_epochs 100 --samples_per_class 1000 --layerwise True --jobname $DATE
    python dataset_transform.py --pretrain_model RawCnn --data_path t1 --model_data_path default --seed $SEED --n_epochs 100 --samples_per_class 1000 --layerwise True --jobname $DATE
done

# 4. Train the probing classifiers
echo "Training probes..."
for SEED in 6 7 8 9 10; do
    python train.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs 100 --samples_per_class 1000 --jobname $DATE
done

# 5. Run Evaluations (T1, T3, T4)
echo "Running unit tests T1, T3, T4..."
for SEED in 6 7 8 9 10; do
    # T1: Groundedness test
    python t1.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs 100 --samples_per_class 1000 --jobname $DATE
    
    # T3/T4: Modularity and Causal tests (uses INLP)
    python t3t4.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs 100 --samples_per_class 1000 --jobname $DATE
done

echo "Done! Results are in results/${DATE}/"
