#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=15:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G   # memory per CPU core
#SBATCH -J "nih_effnetV2L"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/groups/CS156b/conda_installs/ups_conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/groups/CS156b/conda_installs/ups_conda/etc/profile.d/conda.sh" ]; then
        . "/groups/CS156b/conda_installs/ups_conda/etc/profile.d/conda.sh"
    else
        export PATH="/groups/CS156b/conda_installs/ups_conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate 310

module load cuda/11.2

cd /groups/CS156b/teams/ms/unemployed_pony_spuds/main

python run_baseline.py --epochs 3 --batch_size 16 --num_workers 15 --lr_patience 5 --custom_save nih_effnetV2L \
--model_type EfficientNetV2L --restore_best_weights \
--dataset_type nih --img_size 456 --dataset_dir /groups/CS156b/teams/ups_data/nih_456 --multi_gpu