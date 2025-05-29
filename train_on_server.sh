#!/bin/bash
#SBATCH --job-name=hsv_unet_train
#SBATCH --output=train_output_%j.log
#SBATCH --error=train_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Load necessary modules (adjust these based on your server's configuration)
module load cuda/11.8
module load python/3.9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the training script
python Src/train_unet_hsv.py 