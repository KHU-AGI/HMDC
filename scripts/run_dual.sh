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
model_2=${8}
lr_net_2=${9}
path=${10}

cmd=("" "--feature_matching_in_training" "--feature_matching_in_condensation" "--gradient_accumulation" "--gradient_scale_and_clip")

echo ${cmd[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=12
export HF_HOME=/data/dlwogh9344/.cache/huggingface
seeds=(158 159 700) 
ngpu=${CUDA_VISIBLE_DEVICES//,/ } # Remove commas
ngpu=( $ngpu ) # Create shell array
ngpu=${#ngpu[@]}
##### reproduce dc results#########
for seed in ${seeds[@]}
do
       echo "seed: ${seed}"
       python main.py \
       --method DualCondensation \
       --dataset ${dataset} \
       --model ${model} \
       --model_2 ${model_2} \
       --init ${init} \
       --dsa \
       --n_ipc ${n_ipc} \
       --n_data 2000 \
       --iteration 100 \
       --epochs 20 \
       --o_iter 100 \
       --i_iter 1 \
       --lr_img ${lr_img} \
       --lr_net ${lr_net} \
       --lr_net_2 ${lr_net_2} \
       --seed ${seed} \
       --batchsize ${batchsize} \
       --path "./${path}_${seed}" \
       --data_path ./data
done