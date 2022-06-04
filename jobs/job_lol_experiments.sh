#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=20:00:00   # walltime
#SBATCH --ntasks=15   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -J "lol_experiments"   # job name

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

cd /home/knakamur/unemployed_pony_spuds/jobs
cd ../main
CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 5 --batch_size 128 --num_workers 15 --model_type EfficientNetB0 --lr_patience 100 \
--save_prefix lol --save_postfix norm

CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 5 --batch_size 128 --num_workers 15 --model_type EfficientNetB0 --lr_patience 100 \
--loss_func weighted \
--save_prefix lol --save_postfix weighted_loss

CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 5 --batch_size 128 --num_workers 15 --model_type EfficientNetB0 --lr_patience 100 \
--generator aug \
--save_prefix lol --save_postfix aug

CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 5 --batch_size 128 --num_workers 15 --model_type EfficientNetB0 --lr_patience 100 \
--data_mode frontal \
--save_prefix lol --save_postfix frontal

CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 5 --batch_size 128 --num_workers 15 --model_type EfficientNetB0 --lr_patience 100 \
--data_mode lateral \
--save_prefix lol --save_postfix lateral