#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=9:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "unet_baseline_gpu"   # job name

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

module load cuda/11.0

cd /home/jyhuang/unemployed_pony_spuds-1
cd main
CUDA_VISIBLE_DEVICES=0 python run_baseline.py --epochs 10 --batch_size 128 --num_workers 10 --lr_patience 5 --save_postfix unfrozen --model_type EfficientNetB4 --dataset_dir /central/groups/CS156b/data