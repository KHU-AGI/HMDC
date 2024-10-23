#!/bin/bash

conda --version
python --version

dataset=${1}
model=${2}
batchsize=${3}
lr_img=${4}
lr_net=${5}
path=${6}

##### reproduce dc results#########
python main.py \
       --method Continual \
       --dataset ${dataset} \
       --model ${model} \
       --init aug_mean \
       --dsa \
       --num_task 5 \
       --epochs 5 \
       --seed 158 \
       --n_ipc 10 \
       --Iteration 50 \
       --o_iter 100 \
       --i_iter 1000 \
       --batchsize ${batchsize} \
       --lambda_1 0.05 \
       --lambda_2 0.05 \
       --lr_img ${lr_img} \
       --lr_net ${lr_net} \
       --path ./$path
# python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode Mㄴㅁㅇㄹ