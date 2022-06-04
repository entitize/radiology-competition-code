#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=5:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "preprocess_datasetup"   # job name
#SBATCH --array=0-10
#SBATCH --mail-user=mshao@caltech.edu
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

cd /home/knakamur/unemployed_pony_spuds/dataset_setup
# cd /home/mshao/unemployed_pony_spuds/dataset_setup

python preprocess_datasetup.py --dest_dir=preprocessed_original