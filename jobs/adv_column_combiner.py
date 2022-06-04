""" Combines the disease columns, if specified --skip_missing it will replace missing columns with 1 """
import pandas as pd
import argparse
from icecream import ic
import os
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--type', type=str, choices=['TEST', 'SOLN', 'TEST_ENSEMBLE', 'SOLN_ENSEMBLE'], default='TEST_ENSEMBLE')
parser.add_argument('--skip_missing', action='store_true', default=False)

args = parser.parse_args()

results_dir = "../results"
# config_file = adv_launcher_tests/adv1.json
# simple_dir_name = "adv1_launcher_tests"
# simple_file_name = "adv1"
simple_file_name = args.config_file.split("/")[-1].split(".")[0]
simple_dir_name = "/".join(args.config_file.split("/")[0:-1])

# results are stored here: results/adv_launcher_tests/adv1/adv1_{DISEASE}_TEST_ENSEMBLE.csv
# prefix = results/adv_launcher_tests/adv1/adv1_
# postfix = _TEST_ENSEMBLE.csv
prefix = results_dir + "/" + simple_dir_name + "/" + simple_file_name + "/" + simple_file_name + "_"
postfix = "_" + args.type + ".csv"

diseases = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

# test_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/224_data/student_labels/test_ids.csv")
# soln_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/224_data/student_labels/solution_ids.csv")
test_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/s/EfficientNetV2B3_ALL_FOLDS_TEST_ENSEMBLE_COMBINED.csv")
soln_ids_csv = pd.read_csv("/groups/CS156b/teams/ups_data/s/nice2.csv")

dfs = []
skipped_files = []
for i in range(0, 14):
    file = prefix + str(i) + postfix
    # check if file exists
    if args.skip_missing and not os.path.isfile(file):
        logger.warning(f"Skipping {file}")
        skipped_files.append(file)
        if args.type == "TEST" or args.type == "TEST_ENSEMBLE":
            df = test_ids_csv.copy()
        elif args.type == "SOLN" or args.type == "SOLN_ENSEMBLE":
            df = soln_ids_csv.copy()
        # only select the disease column
        # df.drop("Path", axis=1, inplace=True)
        df = df[[diseases[i]]]
        dfs.append(df)

    elif not os.path.isfile(file):
        logger.error(f"File {file} does not exist")
        logger.info("This may be because one of the categories jobs failed to run or the job hasn't finished yet.")
        logger.info("If the job failed and it was b/c of unexpected cuda error, you can rerun it with the --relaunch_failed flag for the script advanced_launcher.py")
        logger.info("If you want to just want to ignore missing columns, then run this script with --skip_missing. It fills in the missing columns with nice.csv or EfficientNetV2B3_ALL_FOLDS_TEST_ENSEMBLE_COMBINED.csv")
        assert False, f"File {file} does not exist"
    else:
        df = pd.read_csv(file)
        if i != 0:
            df = df.drop(df.columns[0], axis=1)
        logger.success(f"Loaded {file}, shape: {df.shape}")
        dfs.append(df) 


output_file_path = results_dir + "/" + simple_dir_name + "/" + simple_file_name + "_" + args.type + "_COMBINED.csv"
logger.info(f"Skipped {len(skipped_files)} files")
logger.success(f"Saving to {output_file_path}")
combined_dfs = pd.concat(dfs, axis=1)
# set id column
if args.type == "TEST" or args.type == "TEST_ENSEMBLE":
    combined_dfs["Id"] = test_ids_csv["Id"]
elif args.type == "SOLN" or args.type == "SOLN_ENSEMBLE":
    combined_dfs["Id"] = soln_ids_csv["Id"]
combined_dfs = combined_dfs[["Id"] + [col for col in combined_dfs.columns if col != "Id"]]
combined_dfs.to_csv(output_file_path, index=False)

    



