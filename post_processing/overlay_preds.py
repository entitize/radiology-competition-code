""" Overlays predictions on top of some standard csv file. 
Useful for models that only predict one class"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_preds_csv', type=str, default="sanity_base_1.csv")
parser.add_argument('--overlay_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

results_dir = "../results"

args = parser.parse_args()

df = pd.read_csv(results_dir + "/" + args.base_preds_csv)
top_df = pd.read_csv(results_dir + "/" + args.overlay_file)

# replace the columns of df with the top_df
for col in top_df.columns:
    if col in df.columns:
        print(f"Replacing {col}")
        df[col] = top_df[col]

save_file = results_dir + "/" + args.output_file
df.to_csv(save_file, index=False)
print(f"Saved to {save_file}")