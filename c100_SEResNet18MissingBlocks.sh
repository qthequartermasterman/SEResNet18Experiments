#!/bin/bash
#BATCH --job-name=SEResNet18
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrewsansom@my.unt.edu
#SBATCH --ntasks=1
#SBATCH --qos=large
#SBATCH -p public
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH -t 500:00:00
#SBATCH --output=outlog/out_%j.log
module load python
module load cuda/75/blas/7.5.18
module load cuda/75/fft/7.5.18
module load cuda/75/nsight/7.5.18
module load cuda/75/profiler/7.5.18
module load cuda/75/toolkit/7.5.18
module load cudnn/6.0/cuda75
module load pytorch/1.1.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./cifar.py --netName=SEResNet18WithOneLayerOfSEBlock --bs=512 --cifar=100 --missing=10
