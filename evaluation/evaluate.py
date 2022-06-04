""" Evaluate a model's performance on the evaluation set """
import pandas as pd
import numpy as np
import argparse
from helpers import helper

results_dir = "../results"

# take in an eval set csv file 
parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", type=str, required=True)

args = parser.parse_args()

h = helper.DataHelper(args)

# TODO: Visualizations

# ROU AUC visualization

# Confusion Matrix

# Accuracy per category

# etc.