import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

# arg parse take in a list of files
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs='+', required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--use_absolute", action='store_true', default=False)
parser.add_argument("--base_dir", type=str, default='../results/')

args = parser.parse_args()

results_dir = "../results/"

# read each of the input_files as csv
dfs = []
for input_file in args.input_files:
    if not args.use_absolute:
        df = pd.read_csv(args.base_dir + input_file)
    else: 
        df = pd.read_csv(input_file)
    dfs.append(df)

# check that each df has the same size
for df in dfs:
    assert df.shape[0] == dfs[0].shape[0]

# compute a final df that contains the average of the probabilities for each cell
final_df = pd.DataFrame(index=dfs[0].index, columns=dfs[0].columns)
# for df in dfs:
#     final_df += df
# final_df /= len(dfs)
for col in tqdm(dfs[0].columns):
    if col == dfs[0].columns[0]:
        final_df[col] = dfs[0][col]
        continue
    for row in dfs[0].index:
        probs = []
        for df in dfs:
            probs.append(df.loc[row, col])
        final_df.loc[row, col] = np.mean(probs)

final_df.to_csv(results_dir + args.output_file, index=False)
logger.success(f"Saved to {results_dir + args.output_file}")