#!/bin/sh

#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

nvidia-smi
echo $CUDA_VISIBLE_DEVICES
source /home/s1116548/anaconda3/bin/activate 
#conda activate hfc


python train.py --cv ${cv} --epoch 20
