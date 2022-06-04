""" Automatically combines the results of k-folds and generates both test and solution csv files. """
import loguru
import pandas as pd
import argparse
from icecream import ic
import os
from loguru import logger
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--skip_missing', action='store_true', default=False)
parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
args = parser.parse_args()

simple_file_name = args.config_file.split("/")[-1].split(".")[0]
simple_dir_name = "/".join(args.config_file.split("/")[0:-1])

num_folds = 5
file_prefix = simple_file_name[:-1]

the_folds = args.folds

for k in the_folds:
    if args.skip_missing:
        call(["python", "adv_column_combiner.py", f"--config_file={simple_dir_name}/{file_prefix}{k}.json", "--type=TEST_ENSEMBLE", "--skip_missing"])
        call(["python", "adv_column_combiner.py", f"--config_file={simple_dir_name}/{file_prefix}{k}.json", "--type=SOLN_ENSEMBLE", "--skip_missing"])
    else:
        call(["python", "adv_column_combiner.py", f"--config_file={simple_dir_name}/{file_prefix}{k}.json", "--type=TEST_ENSEMBLE"])
        call(["python", "adv_column_combiner.py", f"--config_file={simple_dir_name}/{file_prefix}{k}.json", "--type=SOLN_ENSEMBLE"])

# python take_avg_probs_results.py --input_files k_fold_exps/DenseNet169_k0_TEST_ENSEMBLE_COMBINED.csv k_fold_exps/DenseNet169_k1_TEST_ENSEMBLE_COMBINED.csv k_fold_exps/DenseNet169_k3_TEST_ENSEMBLE_COMBINED.csv k_fold_exps/DenseNet169_k4_TEST_ENSEMBLE_COMBINED.csv --output_file k_fold_exps/DenseNet169_ALL_FOLDS_TEST_ENSEMBLE_COMBINED.csv
test_results_files = []
soln_results_files = []

for k in the_folds:
    test_results_files.append(f"{simple_dir_name}/{file_prefix}{k}_TEST_ENSEMBLE_COMBINED.csv")
    soln_results_files.append(f"{simple_dir_name}/{file_prefix}{k}_SOLN_ENSEMBLE_COMBINED.csv")

logger.debug(f"test_results_files: {test_results_files}")
logger.debug(f"soln_results_files: {soln_results_files}")
call(["python", "../post_processing/take_avg_probs_results.py", "--input_files"] + test_results_files + ["--output_file", f"{simple_dir_name}/{file_prefix}_ALL_FOLDS_TEST_ENSEMBLE_COMBINED.csv"])
call(["python", "../post_processing/take_avg_probs_results.py", "--input_files"] + soln_results_files + ["--output_file", f"{simple_dir_name}/{file_prefix}_ALL_FOLDS_SOLN_ENSEMBLE_COMBINED.csv"])

logger.success(f"Final test results file: ../results/{simple_dir_name}/{file_prefix}_ALL_FOLDS_TEST_ENSEMBLE_COMBINED.csv")
logger.success(f"Final soln results file: ../results/{simple_dir_name}/{file_prefix}_ALL_FOLDS_SOLN_ENSEMBLE_COMBINED.csv")
logger.info("Pro tip: hold command and click file name above to jump to the file")