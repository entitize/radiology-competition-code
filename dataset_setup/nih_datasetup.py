# %%

import hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from helpers.models import get_model
from tqdm import tqdm
import cv2
from icecream import ic
import os
import pandas as pd
import argparse
# %%
ds = hub.load('hub://activeloop/nih-chest-xray-train')
raw_dataloader = ds.tensorflow().prefetch(tf.data.AUTOTUNE)

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dest_dir", type=str, required=True)
parser.add_argument('--res', type=int, default=224)
parser.add_argument("--test_mode", action="store_true", default=False)
args = parser.parse_args()

# %%
dest_dir = f"/groups/CS156b/teams/ups_data/{args.dest_dir}"
images_save_dir = f"{dest_dir}/images"

# if directory doesn't exist, create it
if not os.path.exists(images_save_dir):
    os.makedirs(images_save_dir)

def process_data(ds):
    labels = []
    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        img = sample['images'].numpy()
        findings = sample['findings']
        findings = tf.cast(findings, tf.int32)
        y = tf.one_hot(findings, 14)
        y = tf.reduce_sum(y, axis=0)
        row = y.numpy().tolist()
        img = cv2.resize(img, (args.res, args.res))

        # save image to disk
        file_name = f"{i}.png"
        file_path = f"{images_save_dir}/{file_name}"
        cv2.imwrite(file_path, img)

        row = [file_name] + row
        labels.append(row)

        if args.test_mode and i > 50:
            break
    # save labels to csv
    columns = ['Path', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14']
    df = pd.DataFrame(labels, columns=columns)
    df.to_csv(f"{dest_dir}/train.csv", index=False)
    print(f"Saved {len(labels)} images to {images_save_dir}")

process_data(ds)