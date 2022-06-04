""" Saves augmented images in directory 
saves preprocessing time in future """
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os
import cv2
import argparse
from icecream import ic

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="/groups/CS156b/data")
parser.add_argument("--dest_dir", type=str, required=True)
parser.add_argument("--test_mode", action="store_true", default=False)
args = parser.parse_args()

data_dir_path = args.source_dir
train_df_path = args.source_dir + "/student_labels/train.csv"
test_df_path = args.source_dir + "/student_labels/test_ids.csv"
dest_dir = f"/groups/CS156b/teams/ups_data/{args.dest_dir}"


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
else:
    assert False, "dest_dir already exists"

ic(f"data_dir_path: {data_dir_path}")
train_df = pd.read_csv(train_df_path)
train_df.head()
test_df = pd.read_csv(test_df_path)
test_df.head()
# %%
all_df = pd.concat([train_df, test_df])

if not args.test_mode:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) # ranges from 0 - 10 inclusive
    portion_size = int(len(all_df) / 10)
    all_df = all_df.iloc[task_id * portion_size : (task_id + 1) * portion_size]
else:
    all_df = all_df.iloc[:50]

# %%
for i, row in tqdm(all_df.iterrows(), total=len(all_df)):
    image_path = data_dir_path + "/" + row["Path"] 
    dest_path = dest_dir + "/" + row["Path"]
    dest_path_dir = "/".join(dest_path.split("/")[:-1])
    if os.path.exists(image_path):
        os.makedirs(dest_path_dir, exist_ok=True)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(dest_path, img)

shutil.copytree(data_dir_path + "/student_labels", dest_dir + "/student_labels")

print(f"Done, saved to {dest_dir}")
