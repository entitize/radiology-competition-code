""" Saves images with a specified resolution from the original dataset into a new directory. 
Saves preprocessing time in future """
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import shutil
import argparse
import os
import cv2
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--only_solution', default=False, action='store_true')
args = parser.parse_args()

# See jobs/224_datasetup/job_224_datasetup.sh for how to run this script
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) # ranges from 0 - 10 inclusive

# %%
data_dir_path = "/groups/CS156b/data"
train_df_path = "/groups/CS156b/data/student_labels/train.csv"
test_df_path = "/groups/CS156b/data/student_labels/test_ids.csv"
soln_df_path = "/groups/CS156b/data/student_labels/solution_ids.csv"
dest_dir = f"/groups/CS156b/teams/ups_data/{args.res}_data"
# %%

train_df = pd.read_csv(train_df_path)
train_df.head()
test_df = pd.read_csv(test_df_path)
test_df.head()
solution_df = pd.read_csv(soln_df_path)
solution_df.head()
# %%
if args.only_solution:
    all_df = solution_df
else:
    all_df = pd.concat([train_df, test_df, solution_df])
portion_size = int(len(all_df) / 10)
all_df = all_df.iloc[task_id * portion_size : (task_id + 1) * portion_size]

# %%
# for each image in the Path column, copy it to the dest_dir
for i, row in tqdm(all_df.iterrows(), total=len(all_df)):
    image_path = data_dir_path + "/" + row["Path"] # e.g. chexpert/test/pid54934/study8/view1_frontal.jpg
    dest_path = dest_dir + "/" + row["Path"] # e.g. dest/chexpert/test/pid54934/study8/view1_frontal.jpg
    dest_path_dir = "/".join(dest_path.split("/")[:-1])
    if os.path.exists(image_path):
        if not os.path.exists(dest_path_dir):
            os.makedirs(dest_path_dir, exist_ok=True)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (args.res, args.res))
        cv2.imwrite(dest_path, img)

if not os.path.exists(dest_dir + "/student_labels"):
    shutil.copytree(data_dir_path + "/student_labels", dest_dir + "/student_labels")

print(f"Done, saved to {dest_dir}")
