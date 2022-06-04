#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=5G   # memory per CPU core
#SBATCH -J "pony6_2"   # job name

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

cd /home/knakamur/unemployed_pony_spuds/main

python run_baseline.py --epochs 20 --batch_size 128 --num_workers 15 --lr_patience 6 --custom_save pony_6_2 \
--model_type EfficientNetB0 --dataset_dir /groups/CS156b/teams/ups_data/224_data --restore_best_weights --monitor val_auc --ensemble_best_epochs 3 \
--lr_reducer_patience 3 --multi_gpu
