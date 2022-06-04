"""Combines the predictions csvs"""
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--input_files", type=str, nargs="+")
parser.add_argument("--output_file", type=str, required=True)

args = parser.parse_args()

results_dir = "../results"

dfs = []
for file in args.input_files:
    df = pd.read_csv(results_dir + "/" + file)
    print(f"Loaded {file}, shape: {df.shape}")
    dfs.append(df)

# Combine the dfs on top of each other
df = pd.concat(dfs)

save_file = f"{results_dir}/{args.output_file}"

df.to_csv(save_file, index=False)
print(f"Saved to {save_file}")
print(f"{df.shape}")