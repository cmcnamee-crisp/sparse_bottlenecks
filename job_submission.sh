#!/bin/bash
# job_submission.sh
# Example SLURM job submission script for the Kempner cluster

#SBATCH --job-name=blah
#SBATCH --partition=seas_gpu
#SBATCH --account=ba_lab
#SBATCH --time=1-00:00:00
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-0
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

# Replace these with your actual details
USERNAME="cmcnamee"
LAB="ba_lab"
FOLDER_NAME="sparse_bottlenecks"
ENV_NAME="sparse_bottlenecks"

# 1. Move to the correct directory
echo "Moving to project directory..."
cd /n/${LAB}/Everyone/${USERNAME}/${FOLDER_NAME}/

# Ensure logs directory exists for SLURM output
mkdir -p logs

# 2. Setup the environment
echo "Loading modules and activating environment..."
# Load python module (this provides conda/mamba)
module load python
# Initialize conda for this non-interactive shell
eval "$(conda shell.bash hook)"
# Activate the environment using the full path
conda activate /n/${LAB}/Everyone/${USERNAME}/conda/envs/${ENV_NAME}

# 3. Run the code
echo "Starting job execution..."
# REPLACE THIS WITH YOUR ACTUAL COMMAND
python train.py

# 4. Cleanup and save results
echo "Job finished. Deactivating environment..."
mamba deactivate

# Move results back to permanent storage (home directory)
echo "Saving results back to home directory..."
rsync -avx /n/${LAB}/Everyone/${USERNAME}/${FOLDER_NAME}/results/ ~/sparse_bottlenecks_results/
echo "Done!"
