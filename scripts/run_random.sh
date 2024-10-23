dataset=${1}
init=${2}
n_ipc=${3}
path=${4}

echo ${cmd[$SLURM_ARRAY_TASK_ID]}

seeds=(158 159 700) 

##### reproduce dc results#########
for seed in ${seeds[@]}
do
       echo "seed: ${seed}"
       echo "  #####################"
       echo "  ${model}"
       echo "  #####################"
       echo ""

       echo ${path[$SLURM_ARRAY_TASK_ID]}

       python main.py \
              --method Random \
              --dataset ${dataset} \
              --model ConvNet \
              --init ${init} \
              --n_ipc ${n_ipc} \
              --seed ${seed} \
              --path ./${path}_${seed}
done