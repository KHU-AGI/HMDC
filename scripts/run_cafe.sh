#!/bin/bash
conda --version
python --version

dataset=${1}
model=${2}
init=${3}
n_ipc=${4}
batchsize=${5}
lr_img=${6}
lr_net=${7}
path=${8}

seeds=(158 159 700) 

##### reproduce dc results#########
for seed in ${seeds[@]}
do
       echo "seed: ${seed}"
       python main.py \
              --seed ${seed} \
              --method CAFE \
              --dataset ${dataset} \
              --model ${model} \
              --init ${init} \
              --dsa \
              --n_ipc ${n_ipc} \
              --iteration 2000 \
              --o_iter 100 \
              --i_iter 1000 \
              --batchsize ${batchsize} \
              --lambda_1 0.05 \
              --lambda_2 0.05 \
              --lr_img ${lr_img} \
              --lr_net ${lr_net} \
              --path "./${path}_${seed}"
done
########cross arch######
# python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M