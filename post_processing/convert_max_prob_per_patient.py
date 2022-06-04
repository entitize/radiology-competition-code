# %%
import pandas as pd
import numpy as np

# Argparse
import argparse

# Accept input file
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)

args = parser.parse_args()
input_file = args.input_file

test_df = pd.read_csv("../data/student_labels/test_ids.csv")

# Extract patient id from path
test_df['patient_id'] = test_df['Path'].str.split('/').str[1]
test_df.head()

# %%
# results file
results_df = pd.read_csv(f"../results/{input_file}")
diseases = list(results_df.columns[-14:])
results_df.head()

# %% Concatenate test_df and results_df on Id column
merged_df = test_df.merge(results_df, on='Id', how='outer')
merged_df = merged_df[["Id"] + ["patient_id"] + diseases]

# %%

# aggregate on id and compute average
agg_df = merged_df.groupby('patient_id').max()
agg_df = agg_df.drop(columns=['Id'])
agg_df.head()
# %%
final_df = pd.merge(test_df, agg_df, on='patient_id', how='inner')
final_df = final_df.drop(columns=['Path', 'patient_id'])
final_df = final_df.set_index('Id')
final_df
# %%
final_df.to_csv(f"../results/postprocessed_results/mmax_{input_file}")
print(f"Saved to ../results/postprocessed_results/mmax_{input_file}")

