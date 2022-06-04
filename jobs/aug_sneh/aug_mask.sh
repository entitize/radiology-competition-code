#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=5:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH -J "mask"   # job name
#SBATCH --array=0-10

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

cd /home/smpatel/unemployed_pony_spuds/dataset_setup

python aug_sneh.py --dest_dir mask