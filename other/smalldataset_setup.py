# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os

# %%
data_dir_path = "/home/knakamur/CS156b/data"
train_df_path = "/home/knakamur/CS156b/data/student_labels/train.csv"
test_df_path = "/home/knakamur/CS156b/data/student_labels/test_ids.csv"
dest_dir = "dest"
num_train_samples = 200
num_test_samples = 50
# %%

train_df = pd.read_csv(train_df_path)
train_df = train_df.sample(num_train_samples, random_state=42)
train_df.head()
# %%
test_df = pd.read_csv(test_df_path)
test_df = test_df.sample(num_test_samples, random_state=42)
test_df.head()
# %%
all_df = pd.concat([train_df, test_df])
all_df.head()

# %%
# for each image in the Path column, copy it to the dest_dir
for i, row in tqdm(all_df.iterrows()):
    image_path = data_dir_path + "/" + row["Path"] # e.g. chexpert/test/pid54934/study8/view1_frontal.jpg
    dest_path = dest_dir + "/" + row["Path"] # e.g. dest/chexpert/test/pid54934/study8/view1_frontal.jpg
    dest_path_dir = "/".join(dest_path.split("/")[:-1])
    if os.path.exists(image_path):
        os.makedirs(dest_path_dir, exist_ok=True)
        shutil.copy(image_path, dest_path)
        # print(image_path, dest_path, dest_path_dir)

# also save the train and test dataframes
train_df.to_csv(dest_dir + "/train.csv", index=False)
test_df.to_csv(dest_dir + "/test_ids.csv", index=False)

print(f"Done, saved to {dest_dir}")
# %%

# %%
