from numpy import absolute
from simple_slurm import Slurm
import argparse
from pathlib import Path
from icecream import ic
import json
import datetime
from loguru import logger
import os

# Instructions:
# 1. IMPORTANT: Make sure your ~/.bashrc file has this: 
"""
# >>> conda initialize >>>
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
"""

# 2. Create a directory to store your configuration in
# 3. Copy the adv_launcher_tests/adv1.json into that created directory. We'll call it DIR
# 4. Rename the file. We'll call it FILE
# 5. Edit the file with your own settings.
# Note: Whatever is in defaults will be overwritten by the specific disease category settings.
# For example, if you define "--model_type": "EfficientNetB0" in defaults but define "--model_type": "ResNet152" in Edema, the model for Edema will be ResNet152.
# Note: Slurm settings go in slurm and python settings go in python
# 6. Run this python file by running "python3 advanced_launcher.py --config_file DIR/FILE"
# 7. Check job status here: https://interactive.hpc.caltech.edu/pun/sys/dashboard/activejobs?jobcluster=all&jobfilter=user
# 8. Use the --relaunch_failed flag to relaunch failed jobs. It scans the .out files and see if it contains a cuda error in which case it will relaunch it.

def get_absolute_path_to_main():
    absolute_path = str(Path(__file__).parent.absolute())
    absolute_path_chunks = absolute_path.split("/")
    absolute_path_chunks.pop()
    absolute_path_chunks.append("main")
    absolute_path = "/".join(absolute_path_chunks)
    return absolute_path
def get_absolute_path_to_post_processing():
    absolute_path = str(Path(__file__).parent.absolute())
    absolute_path_chunks = absolute_path.split("/")
    absolute_path_chunks.pop()
    absolute_path_chunks.append("post_processing")
    absolute_path = "/".join(absolute_path_chunks)
    return absolute_path

def get_absolute_path_to_project_dir():
    absolute_path = str(Path(__file__).parent.absolute())
    absolute_path_chunks = absolute_path.split("/")
    absolute_path_chunks.pop()
    absolute_path = "/".join(absolute_path_chunks)
    return absolute_path

absolute_path_to_main = get_absolute_path_to_main()
absolute_path_to_post_processing = get_absolute_path_to_post_processing()

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--enable_requeue', action='store_true', default=False) # not recommended using, just run this script against with --relaunch_failed flag
parser.add_argument('--relaunch_failed', action='store_true', default=False)
parser.add_argument('--disease_ids', nargs='+', type=int, default=[]) # in order to pass specific diseases, use --disease_ids 1 2 3 4 5
parser.add_argument('--skip_disease_ids', nargs='+', type=int, default=[]) # skips specified diseases, commonly pass 0
args = parser.parse_args()

if args.enable_requeue:
    logger.warning("WARNING: You are enabling the --enable_requeue flag. This flag is not recommended. It is better to run this script again using --relaunch_failed flag after the job fails")

config_file = args.config_file
base_job_name = config_file
all_diseases = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
diseases = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
if args.disease_ids:
    # convert specific diseases ids to disease names
    diseases = [diseases[i] for i in args.disease_ids]
if args.debug:
    diseases = ["No Finding", "Enlarged Cardiomediastinum"]

# read the config json file
with open(config_file) as f:
    config = json.load(f)

def get_category_config(category, python_or_slurm):
    if python_or_slurm == 'python':
        json = config[category]['python']
    elif python_or_slurm == 'slurm':
        json = config[category]['slurm']
    else:
        assert False, "python_or_slurm must be either 'python' or 'slurm'"
    return json

def get_flattened_json_for_python(category):
    assert category != "defaults"
    defaults_config = get_category_config("defaults", 'python')
    category_config = get_category_config(category, 'python')
    # merge defaults_config and category_config. if a key is in both, category_config overrides
    final_config = {}
    for key in defaults_config:
        final_config[key] = defaults_config[key]
    for key in category_config:
        final_config[key] = category_config[key]
    # flatten the json
    res = ""
    for key in final_config:
        res += f"{key} {final_config[key]} "
    # Make sure that --diseases, --custom_save, or --custom_save_dir is not in the final_config
    assert "--diseases" not in res
    assert "--custom_save" not in res
    assert "--custom_save_dir" not in res
    return res

def get_slurm_args(category):
    category_slurm_args = get_category_config(category, "slurm")
    default_slurm_args = get_category_config("defaults", "slurm")
    # merge defaults_config and category_config. if a key is in both, category_config overrides
    final_slurm_args = {}
    for key in default_slurm_args:
        final_slurm_args[key] = default_slurm_args[key]
    for key in category_slurm_args:
        final_slurm_args[key] = category_slurm_args[key]
    return final_slurm_args

# Launching 14 experiments
job_ids = []
for disease in diseases:
    # disease id is the index of the disease in the diseases list
    disease_id = all_diseases.index(disease)
    if disease_id in args.skip_disease_ids:
        continue
    job_name = f"{base_job_name}_{disease_id}"

    if args.relaunch_failed:
        # determine if this job is a failed job
        log_file = f"{get_absolute_path_to_project_dir()}/jobs/{job_name}.out"
        # read log file and check if it contains "cuda error" in any of the lines
        if not os.path.exists(log_file):
            continue
        with open(log_file) as f:
            lines = f.readlines()
        relaunch_this_one = False
        for line in lines:
            if "CUDA_ERROR_UNKNOWN: unknown error" in line:
                relaunch_this_one = True
        if not relaunch_this_one:
            continue
        logger.success(f"Relaunching {job_name}")

    simple_name = base_job_name.split("/")[-1].split(".")[0] + "_" + str(disease_id)
    simple_dir = base_job_name[:-5]

    slurm_args = get_slurm_args(disease)
    command = f"source ~/.bashrc; "
    command += f"module load cuda/11.2; "
    command += f"cd {absolute_path_to_main}; "
    command += f"python run_baseline.py --disease {disease_id} {get_flattened_json_for_python(disease)}"
    command += f"--custom_save_dir {simple_dir} --custom_save {simple_name}"
    if args.enable_requeue:
        command += f" || scontrol requeuehold {Slurm.SLURM_JOB_ID}"

    slurm = Slurm(job_name=job_name, **slurm_args, output=f"{job_name}.out")

    if args.debug:
        print("========= DISEASE: ", disease, " =========")
        print(job_name)
        print(slurm_args)
        print(command)
        continue

    job_id = slurm.sbatch(command)
    job_ids.append(job_id)
    logger.success(f"Launched {job_name} with job id {job_id}")

j_name = base_job_name.split(".")[0]

# write job_ids to advanced_launcher_history.out
with open(f"advanced_launcher_history.out", "a") as f:
    # write current datetime and job name
    f.write("========== " + str(datetime.datetime.now()) + " ==========\n")
    f.write(f"{j_name}: {job_ids}\n")

if len(job_ids) == 0:
    logger.error("No jobs were launched.")
else:
    logger.success(f"Launched {len(job_ids)} jobs.")
    logger.info(f"If you want to cancel, run: scancel {{{job_ids[0]}..{job_ids[-1]}}}")