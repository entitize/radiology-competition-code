""" Saves augmented images in directory
saves preprocessing time in future """
# %%
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import os
import cv2
import argparse
from icecream import ic
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'helpers' )
sys.path.append( mymodule_dir )
import preprocessing

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="/groups/CS156b/data")
parser.add_argument("--dest_dir", type=str, required=True)
parser.add_argument('--res', type=int, default=224)
parser.add_argument("--test_mode", action="store_true", default=False)
# parser.add_argument("--remove_diaphragm", type=bool, default=True)
# parser.add_argument("--filters", type=bool, default=True)
# parser.add_argument("--borders", type=bool, default=True)
# parser.add_argument("--mask", type=bool, default=True)
parser.add_argument("--remove_diaphragm", action="store_true", default=False)
parser.add_argument("--filters", action="store_true", default=False)
parser.add_argument("--borders", action="store_true", default=False)
parser.add_argument("--mask", action="store_true", default=False)
parser.add_argument('--only_solution', default=False, action='store_true')
args = parser.parse_args()

data_dir_path = args.source_dir
train_df_path = args.source_dir + "/student_labels/train.csv"
test_df_path = args.source_dir + "/student_labels/test_ids.csv"
soln_df_path = "/groups/CS156b/data/student_labels/solution_ids.csv"
dest_dir = f"/groups/CS156b/teams/ups_data/{args.dest_dir}"


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    ic(f"Created {dest_dir}")
else:
    ic("NOTE: dest_dir already exists, will override existing images if they exist")
#     assert False, "dest_dir already exists"

ic(f"data_dir_path: {data_dir_path}")
train_df = pd.read_csv(train_df_path)
train_df.head()
test_df = pd.read_csv(test_df_path)
test_df.head()
solution_df = pd.read_csv(soln_df_path)
solution_df.head()
# %%
all_df = pd.concat([train_df, test_df])

if args.test_mode:
  all_df = all_df.iloc[:50]
else:
  task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) # ranges from 0 - 10 inclusive
  if args.only_solution:
      all_df = solution_df
  else:
      all_df = pd.concat([train_df, test_df, solution_df])
  portion_size = int(len(all_df) / 10)
  all_df = all_df.iloc[task_id * portion_size : (task_id + 1) * portion_size]



# %%
for i, row in tqdm(all_df.iterrows(), total=len(all_df)):
    image_path = data_dir_path + "/" + row["Path"]
    dest_path = dest_dir + "/" + row["Path"]
    dest_path_dir = "/".join(dest_path.split("/")[:-1])
    if os.path.exists(image_path):
        os.makedirs(dest_path_dir, exist_ok=True)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (args.res, args.res))
        img = preprocessing.preprocess(img, diaphragm=(args.remove_diaphragm), filters=args.filters, borders=args.borders, mask=args.mask)
        # img = preprocessing.remove_diaphragm(img)
        # img = cv2.bilateralFilter(img, 9, 75, 75)
        # img = cv2.equalizeHist(img)
        cv2.imwrite(dest_path, img)

if not os.path.exists(dest_dir + "/student_labels"):
    shutil.copytree(data_dir_path + "/student_labels", dest_dir + "/student_labels")

print(f"Done, saved to {dest_dir}")
