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


def preprocess_border(img):
        (img_height, img_length, channels) = img.shape
        border_chance = 1
        x = np.random.uniform(0, 1)
        ret = np.copy(img)
        if (x > border_chance):
            return ret
        # bottom_border = np.random.randint((img_height // 2) + 2, img_height - 2)
        # top_border = np.random.randint(2, (img_height // 2) - 2)
        # right_border = np.random.randint((img_length // 2) + 2, img_length - 2)
        # left_border = np.random.randint(2, (img_length // 2) - 2)

        bottom_border = img_height - 10
        top_border = 10
        right_border = img_length - 10
        left_border = 10

        for i in range(img_height):
            for j in range(img_length):
                for k in range(channels):
                    if (i > bottom_border or i < top_border) or (j > right_border or j < left_border):
                        ret[i][j][k] = 1

        return ret

def preprocess_double_crop(img1, img2):
    (img1_height, img1_length, channels) = img1.shape
    crop_chance = 1
    x = np.random.uniform(0, 1)
    ret = np.copy(img1) 
    if (x > crop_chance):
        return ret
    
    crop_width = np.random.randint(img1_height / 3, img1_height / 2)
    crop_length = np.random.randint(img1_height / 3, img1_length / 2)
    init_location_x = np.random.randint(2, img1_length - crop_width - 2)
    init_location_y = np.random.randint(2, img1_height - crop_length - 2)

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                if (i >= init_location_y and i < init_location_y + crop_width) and (j >= init_location_x and j < init_location_x + crop_length):
                    ret[i][j][k] = img2[i][j][k]
    
    return ret

def preprocess_avg(img1, img2):
    (img1_height, img1_length, channels) = img1.shape
    avg_chance = 1
    x = np.random.uniform(0, 1)
    ret = np.copy(img1) 
    if (x > avg_chance):
        return ret

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                ret[i][j][k] = (img2[i][j][k] + img1[i][j][k]) / 2
    
    return ret

def preprocess_mask(img1):
    (img1_height, img1_length, channels) = img1.shape
    mask_chance = 1
    x = np.random.uniform(0, 1)
    ret = np.copy(img1) 
    if (x > mask_chance):
        return ret
    
    mask_width = np.random.randint(img1_height / 3, img1_height / 2)
    mask_length = np.random.randint(img1_height / 3, img1_length / 2)
    init_location_x = np.random.randint(2, img1_length - mask_width - 2)
    init_location_y = np.random.randint(2, img1_height - mask_length - 2)

    for i in range(img1_height):
        for j in range(img1_length):
            for k in range(channels):
                if (i >= init_location_y and i < init_location_y + mask_width) and (j >= init_location_x and j < init_location_x + mask_length):
                    b = ret[i][j][k]
                    ret[i][j][k] = np.random.uniform(0, b)
    
    return ret


def preprocess_all(img1, img2, crop, avg, border, mask):
    ret = img1
    if crop:
        ret = preprocess_double_crop(img1, img2)
    if avg:
        ret = preprocess_avg(img1, img2)
    if border:
        ret = preprocess_border(ret)
    if mask:
        ret = preprocess_mask(ret)
    
    return ret

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="/groups/CS156b/data")
parser.add_argument("--dest_dir", type=str, required=True)
parser.add_argument("--test_mode", action="store_true", default=False)
args = parser.parse_args()

crop_bool = avg_bool = border_bool = mask_bool = False

if args.dest_dir == "crop":
    crop_bool = True
elif args.dest_dir == "avg":
    avg_bool = True
elif args.dest_dir == "border":
    border_bool = True
elif args.dest_dir == "mask":
    mask_bool = True
else:
    assert False, "dest_dir must be crop, avg, border, or mask"

data_dir_path = args.source_dir
train_df_path = args.source_dir + "/student_labels/train.csv"
test_df_path = args.source_dir + "/student_labels/test_ids.csv"
dest_dir = f"/groups/CS156b/teams/ups_data/{args.dest_dir}"


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
# else:
#     assert False, "dest_dir already exists"

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
i_prev = -1

for i, row in tqdm(all_df.iterrows(), total=len(all_df)):
    if i_prev == -1:
        i_prev = i
        row_prev = row
        continue
    image_path_prev = data_dir_path + "/" + row_prev["Path"]
    dest_path_prev = dest_dir + "/" + row_prev["Path"]
    dest_path_dir_prev = "/".join(dest_path_prev.split("/")[:-1])
    image_path = data_dir_path + "/" + row["Path"]
    dest_path = dest_dir + "/" + row["Path"]
    dest_path_dir = "/".join(dest_path.split("/")[:-1])
    if os.path.exists(image_path) and os.path.exists(image_path_prev):
        if not os.path.exists(dest_path_dir_prev):
            os.makedirs(dest_path_dir_prev, exist_ok=True)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img_prev = cv2.imread(image_path_prev)
        img_prev = cv2.resize(img_prev, (224, 224))
        img_new = preprocess_all(img_prev, img, crop_bool, avg_bool, border_bool, mask_bool)
        cv2.imwrite(dest_path_prev, img_new)
    i_prev = i
    row_prev = row


if not os.path.exists(dest_dir + "/student_labels"):
    shutil.copytree(data_dir_path + "/student_labels", dest_dir + "/student_labels")

print(f"Done, saved to {dest_dir}")
