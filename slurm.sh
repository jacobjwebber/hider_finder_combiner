#!/bin/bash

#SBATCH --time=8-01:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --partition=ILCC_GPU_UBUNTU
#SBATCH --cpus-per-task 32
#SBATCH -J hfc_0.07

dataset_dir=/disk/scratch/s1116548/data
echo $dataset_dir
mkdir -p $dataset_dir
rsync -r -u ../data $dataset_dir
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

python --version
. /home/s1116548/miniconda3/bin/activate
conda activate hfc_vc
python --version
python hfc.py dataset.root=$dataset_dir training.wandb=True model.beta=0.07 combiner=fastspeech2

