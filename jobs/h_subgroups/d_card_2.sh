#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G   # memory per CPU core
#SBATCH -J "d_card_2_frontal_lateral"   # job name

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

python run_baseline.py --epochs 5 --batch_size 64 --num_workers 15 --lr_patience 5 --custom_save d_card_2_frontal \
--model_type DenseNet121 --dataset_dir /groups/CS156b/teams/ups_data/preprocessed_no_diaphragm_mask --restore_best_weights \
--diseases 2 --nan_fill_mode remove_nan_rows --ensemble_best_epochs 2 --loss_func weighted-one-class --custom_save_dir h_group --data_mode frontal

python run_baseline.py --epochs 5 --batch_size 64 --num_workers 15 --lr_patience 5 --custom_save d_card_2_lateral \
--model_type DenseNet121 --dataset_dir /groups/CS156b/teams/ups_data/preprocessed_no_diaphragm_mask --restore_best_weights \
--diseases 2 --nan_fill_mode remove_nan_rows --ensemble_best_epochs 2 --loss_func weighted-one-class --custom_save_dir h_group --data_mode lateral