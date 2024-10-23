#!/bin/bash

conda --version
python --version

method=${1}
dataset=${2}
n_ipc=${3}

##### reproduce dc results#########
echo "seed: 158"
python main.py \
       --seed 158 \
       --method ${method}_Computation \
       --dataset ${dataset} \
       --model convnet \
       --init aug_real \
       --dsa \
       --n_ipc ${n_ipc} \
       --iteration 1 \
       --o_iter 1 \
       --i_iter 1 \
       --batchsize 2 \
       --lr_img 5e-3 \
       --lr_net 0.01 \
       --model_2 ViT_tiny_ft \
       --lr_net_2 0.001 \
       --path ./${SLURM_JOB_ID}_${seed}