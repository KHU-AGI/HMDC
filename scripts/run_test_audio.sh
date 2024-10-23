#!/bin/bash

# Check for --ft or --scratch in path

dataset=${1}
epochs=${2}
batchsize=${3}
path=("${@:4}")

postfix=""
if [[ "${path[*]}" == *"--ft"* ]]; then
       postfix="_ft"
       path=("${path[@]/"--ft"/}")
fi
if [[ "${path[*]}" == *"--scratch"* ]]; then
       postfix="_scratch"
       path=("${path[@]/"--scratch"/}")
fi

date
ulimit -n 65536

conda --version
python --version

model_list=()
learning_rate=()

# ConvNet_ft
model_list+=("audio_cnn")
learning_rate+=("1e-3")

#ViT_base_ft
model_list+=("hubert"$postfix)
learning_rate+=("1e-5")

seeds=(158 159 700)
echo model_list: ${model_list[@]}
for pth in ${path[@]}
do
       echo "#####################"
       echo ${pth}
       echo "#####################"
       for ((i=0; i<${#model_list}; i++)) do
              echo "  #####################"
              echo "  ${model_list[$i]}"
              echo "  #####################"
              echo ""
              python main.py \
                     --method Test_Audio \
                     --dsa \
                     --seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
                     --dataset ${dataset} \
                     --model ${model_list[$i]} \
                     --epochs ${epochs} \
                     --batchsize ${batchsize} \
                     --lr_net ${learning_rate[$i]} \
                     --path ${pth}
                     # --path ${path}
       done
done

########cross arch######
# python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M