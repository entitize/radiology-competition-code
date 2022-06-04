#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=15:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G   # memory per CPU core
#SBATCH -J "pony5_1_EfficientNetB1_kf4"   # job name

#SBATCH --mail-user=jyhuang@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

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

cd /groups/CS156b/teams/jyhuang/unemployed_pony_spuds
cd main
python run_baseline.py --epochs 5 --use_k_folds --k 4 --lr_reducer_patience 1 --batch_size 64 --num_workers 15 --learning_rate 3e-4 --lr_patience 5 --save_prefix pony5 --save_postfix k4 --model_type EfficientNetB1 --dataset_dir /groups/CS156b/teams/ups_data/240_data