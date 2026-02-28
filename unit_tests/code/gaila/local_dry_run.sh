#!/bin/bash
# local_dry_run.sh
# Tests the entire pipeline end-to-end to catch argument/syntax bugs
# Uses only 1 seed, 2 training samples, and 1 epoch

export WANDB_MODE=disabled

echo "Starting local dry-run pipeline test..."
echo "Setting up local virtual environment..."
python3 -m venv venv_test
source venv_test/bin/activate
pip install --upgrade pip --quiet
pip install torch torchvision ftfy regex tqdm pytorch-lightning pandas seaborn pytest matplotlib wandb scipy scikit-learn --quiet
pip install git+https://github.com/openai/CLIP.git --quiet

# 1. Setup required directories
echo "Setting up output directories..."
bash setup.sh

SEED=6
# N_SAMPLES: enough to have data for all concepts
N_SAMPLES=100
# EPOCHS: run for just 1 epoch instead of 100
EPOCHS=1
# BATCH_SIZE: prevent batch empty errors on tiny test datasets
BATCH_SIZE=2

DATE="local_test"

# 2. Generate Dataset for "default" and "t1"
echo "Generating tiny synthetic datasets..."
python dataset.py --data_path default --seed $SEED --n_train $N_SAMPLES --n_test $N_SAMPLES
python dataset.py --data_path t1 --seed $SEED --n_train $N_SAMPLES --n_test $N_SAMPLES

# 3. Pretrain the CNN (RawCnn)
echo "Pretraining RawCnn (1 epoch)..."
python pretrain.py --pretrain_model RawCnn --data_path default --seed $SEED --n_epochs $EPOCHS --dataspec classes --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --jobname $DATE

# 3.5. Transform Dataset
echo "Transforming dataset..."
python dataset_transform.py --pretrain_model RawCnn --data_path default --seed $SEED --n_epochs $EPOCHS --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --layerwise True --jobname $DATE
python dataset_transform.py --pretrain_model RawCnn --data_path t1 --model_data_path default --seed $SEED --n_epochs $EPOCHS --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --layerwise True --jobname $DATE

# 4. Train the probing classifiers
echo "Training probes (1 epoch)..."
python train.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs $EPOCHS --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --jobname $DATE

# 5. Run Evaluations (T1, T3, T4)
echo "Running unit tests T1, T3, T4..."
# T1: Groundedness test
python t1.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs $EPOCHS --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --jobname $DATE
# T3/T4: Modularity and Causal tests (uses INLP)
python t3t4.py --pretrain_model RawCnn --finetune_model linear --data_path default --seed $SEED --n_epochs $EPOCHS --batch_size $BATCH_SIZE --samples_per_class $N_SAMPLES --jobname $DATE

echo "Dry run complete! If you reached this point, the pipeline executed successfully."
