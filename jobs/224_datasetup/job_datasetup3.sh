#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A CS156b

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "all_soln_datasetup"   # job name
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

cd /home/knakamur/unemployed_pony_spuds/
cd dataset_setup

# python datasetup.py --res 240 --only_solution
# python datasetup.py --res 260 --only_solution
# python datasetup.py --res 299 --only_solution
# python datasetup.py --res 300 --only_solution
# python datasetup.py --res 380 --only_solution
# python datasetup.py --res 384 --only_solution
python datasetup.py --res 456
# python datasetup.py --res 480 --only_solution
# python datasetup.py --res 528 --only_solution
# python datasetup.py --res 576 --only_solution