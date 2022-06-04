"""Combines the predprinttions csvs via columns"""
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
import os
from loguru import logger

parser.add_argument("--input_prefix", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--input_postfix", type=str, default="")
parser.add_argument('--skip_missing', action='store_true', default=False)

args = parser.parse_args()

results_dir = "../results"

prefix = results_dir + "/" + args.input_prefix  
postfix = args.input_postfix + ".csv"

dfs = []
skipped_files = []
diseases = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

test_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/224_data/student_labels/test_ids.csv")
soln_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/224_data/student_labels/solution_ids.csv")

for i in range(0, 14):
    file = prefix + str(i) + postfix
    # check if file exists
    if args.skip_missing and not os.path.isfile(file):
        logger.warning(f"Skipping {file}")
        skipped_files.append(file)
        df = test_ids_csv.copy()
        df.drop("Path", axis=1, inplace=True)
        df[diseases[i]] = 1
        dfs.append(df)

    elif not os.path.isfile(file):
        logger.error(f"File {file} does not exist")
        logger.info("This may be because one of the categories jobs failed to run or the job hasn't finished yet.")
        logger.info("If the job failed and it was b/c of unexpected cuda error, you can rerun it with the --relaunch_failed flag for the script advanced_launcher.py")
        logger.info("If you want to just want to ignore missing columns, then run this script with --skip_missing. It fills in the missing columns with 1s.")
        assert False, f"File {file} does not exist"
    else:
        df = pd.read_csv(file)
        if i != 0:
            df = df.drop(df.columns[0], axis=1)
        logger.success(f"Loaded {file}, shape: {df.shape}")
        dfs.append(df) 

# combine dfs column wise
df = pd.concat(dfs, axis=1)

save_file = f"{results_dir}/{args.output_file}"

df.to_csv(save_file, index=False)

print(f"Saved to {save_file}")
print(f"{df.shape}")
print(f"Skipped files: {len(skipped_files)}")
# if all were skipped then report