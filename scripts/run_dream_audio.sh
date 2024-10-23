#!/bin/bash
conda --version
python --version

dataset=${1}
init=${2}
n_ipc=${3}
batchsize=${4}
lr_img=${5}
model=${6}
lr_net=${7}
path=${8}

echo ${cmd[$SLURM_ARRAY_TASK_ID]}

seeds=(158 159 700) 

##### reproduce dc results#########
for seed in ${seeds[@]}
do
       echo "seed: ${seed}"
       python main.py \
              --seed ${seed} \
              --method DREAM_Audio \
              --dataset ${dataset} \
              --model ${model} \
              --init ${init} \
              --dsa \
              --n_ipc ${n_ipc} \
              --n_data 2000 \
              --iteration 1200 \
              --o_iter 100 \
              --i_iter 1 \
              --batchsize ${batchsize} \
              --lr_img ${lr_img} \
              --lr_net ${lr_net} \
              --path "./${path}_${seed}" \
              --data_path ./data
done