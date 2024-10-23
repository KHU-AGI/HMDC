#!/bin/bash

conda --version
python --version

dataset=${1}
model=${2}
batchsize=${3}
epochs=${4}
num_task=${5}
mem_size=${6}
lr_net=${7}
path=${8}

seeds=(158 159 700) 
##### reproduce dc results#########
python memory_continual.py \
       --dataset ${dataset} \
       --seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
       --model ${model} \
       --num_task ${num_task} \
       --epochs ${epochs} \
       --memory_size ${mem_size} \
       --batchsize ${batchsize} \
       --lr_net ${lr_net} \

########cross arch######
# python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M