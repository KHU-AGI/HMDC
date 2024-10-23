#!/bin/bash

#SBATCH -J TestModel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=4-0
#SBATCH -o %x_%j_%a.out
#SBATCH -e %x_%j_%a.err

dataset=${1}

date
ulimit -n 65536

conda --version
python --version

postfix=""
if [[ "${path[*]}" == *"--ft"* ]]; then
       postfix="_ft"
       path=("${path[@]/"--ft"/}")
fi
if [[ "${path[*]}" == *"--scratch"* ]]; then
       postfix="_scratch"
       path=("${path[@]/"--scratch"/}")
fi

model_list=()
learning_rate=()

# ConvNet_ft
model_list+=("ConvNet")
learning_rate+=("1e-2")

#ResNet18_ft
model_list+=("ResNet18"$postfix)
learning_rate+=("1e-3")

#ResNet50_ft
model_list+=("ResNet50"$postfix)
learning_rate+=("1e-3")

#ResNet101_ft
model_list+=("ResNet101"$postfix)
learning_rate+=("1e-4")

#ViT_small_ft
model_list+=("ViT_tiny"$postfix)
learning_rate+=("1e-3")

#ViT_small_ft
model_list+=("ViT_small"$postfix)
learning_rate+=("1e-3")

#ViT_base_ft
model_list+=("ViT_base"$postfix)
learning_rate+=("1e-4")

length=${#model_list[@]}
seeds=(158 159 700)
for i in $(seq 0 $((length-1)))
do
       ##### reproduce dc results#########
       python main.py \
              --method Upperbound \
              --dataset ${dataset} \
              --model ${model_list[i]} \
              --epochs 2000 \
              --dsa \
              --batchsize 64 \
              --seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
              --lr_net ${learning_rate[i]}
done

########cross arch######
# python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M