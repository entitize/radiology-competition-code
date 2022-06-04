from email.policy import default
from random import choices
import pandas as pd
import tensorflow as tf
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import json
import numpy as np
from keras import backend as K
from icecream import ic


def add_base_parser_args(parser):
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset_dir", type=str, default="../data")
    parser.add_argument("--results_dir", type=str, default="../results")

    parser.add_argument("--batch_size", type=int, default=32, help="Per GPU")
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--num_test_samples", type=int, default=None)

    parser.add_argument("--train_test_split", type=float, default=0.8)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--freeze_base', default=False, action='store_true')
    parser.add_argument('--feature_extraction', default=False, action='store_true')

    parser.add_argument('--save_prefix', type=str, default="base")
    parser.add_argument('--save_postfix', type=str, default="")
    parser.add_argument('--custom_save', type=str, default="")

    parser.add_argument('--log_dir', type=str, default="../logs")

    parser.add_argument('--img_size', type=int, default=None)

    parser.add_argument('--lr_patience', type=int, default=2)

    parser.add_argument('--data_mode', type=str, default="all")


    parser.add_argument('--generator', type=str, default="none", choices=['none', 'aug'])
    parser.add_argument('--diseases', type=str, default="all", choices=['all', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 'g1', 'g2', 'g3', 'g4', 'h2'])

    parser.add_argument('--dataset_type', type=str, default="chexpert", choices=['chexpert', 'nih'])

    parser.add_argument('--debug_soln_set', default=False, action='store_true')

    parser.add_argument('--k', type=int, default=0, help="k-fold cross validation; number between 0-4 inclusive")
    parser.add_argument('--num_folds', type=int, default=5, help="k-fold cross validation")
    parser.add_argument('--use_k_folds', default=False, action='store_true')

class DataHelper():
    def __init__(self, args):

        self.data_dir = args.dataset_dir
        self.models_dir = '../models'
        self.results_dir = '../results'

        if args.dataset_type == 'chexpert':
            self.setup_chexpert_data(args)
        elif args.dataset_type == 'nih':
            self.setup_nih_data(args)

    def setup_chexpert_data(self, args):
        self.train_valid_df = pd.read_csv(f"{self.data_dir}/student_labels/train.csv")
        self.train_valid_df.rename(columns={self.train_valid_df.columns[0]: "Id"}, inplace=True)
        
        if args.diseases == 'all':
            self.diseases = list(self.train_valid_df.columns[-14:])
        elif args.diseases == 'g1':
            self.diseases = ["Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]
        elif args.diseases == 'g2':
            self.diseases = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Pleural Effusion", "Pleural Other", "Pneumothorax"]
        elif args.diseases == 'g3':
            self.diseases = ["No Finding", "Support Devices", "Fracture"]
        elif args.diseases == 'g4':
            self.diseases = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Pleural Effusion", "Pleural Other", "Pneumothorax", "No Finding", "Support Devices", "Fracture"]
        elif args.diseases == 'h2':
            # everything except cardiomelogy and consolidation
            self.diseases = ["No Finding", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion", "Edema", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
        else:
            self.diseases = list(self.train_valid_df.columns[-14:])
            self.diseases = [self.diseases[int(args.diseases)]]

        ic(self.diseases)

        self.batch_size = args.batch_size

        # Getting rid of last row b/c it is bad
        self.train_valid_df = self.train_valid_df.drop(self.train_valid_df.index[-1])
        self.test_df = pd.read_csv(f"{self.data_dir}/student_labels/test_ids.csv")

        if args.debug_soln_set:
            self.soln_df = pd.read_csv(f"{self.data_dir}/student_labels/test_ids.csv").sample(n=5)
        else:
            self.soln_df = pd.read_csv(f"{self.data_dir}/student_labels/solution_ids.csv")

        self.data_mode = args.data_mode
        self.setup_data_mode()

        if args.num_train_samples is not None:
            self.train_valid_df = self.train_valid_df.sample(args.num_train_samples, random_state=1)
            print(f"Using limited: {len(self.train_valid_df)} training/validation samples")
        else:
            print(f"Using full: {len(self.train_valid_df)} training/validation samples")
        if args.num_test_samples is not None:
            self.test_df = self.test_df.sample(args.num_test_samples, random_state=1)
            print(f"Using limited {len(self.test_df)} test samples")
        else:
            print(f"Using full {len(self.test_df)} test samples")

        self.train_valid_df["patient_id"] = self.train_valid_df["Path"].apply(lambda x: x.split("/")[1])

        # Splitting to train and validation based on patient_id
        train_valid_pids = self.train_valid_df["patient_id"].unique()
        
        if args.use_k_folds:
            total_pids = len(train_valid_pids)
            # get the kth fold of pids
            k = args.k
            kth_fold_pids = train_valid_pids[k * total_pids // args.num_folds : (k + 1) * total_pids // args.num_folds]
            train_pids = [pid for pid in train_valid_pids if pid not in kth_fold_pids]
            valid_pids = kth_fold_pids
        else:
            train_pids = train_valid_pids[:int(len(train_valid_pids) * args.train_test_split)]
            valid_pids = train_valid_pids[int(len(train_valid_pids) * args.train_test_split):]

        self.train_df = self.train_valid_df[self.train_valid_df["patient_id"].isin(train_pids)]
        self.valid_df = self.train_valid_df[self.train_valid_df["patient_id"].isin(valid_pids)]


    def setup_nih_data(self, args):
        self.train_valid_df = pd.read_csv(f"{self.data_dir}/train.csv")
        self.train_valid_df["Path"] = self.train_valid_df["Path"].apply(lambda x: f"{self.data_dir}/images/{x}")
        self.train_df = self.train_valid_df.sample(frac=args.train_test_split, random_state=1)
        self.valid_df = self.train_valid_df.drop(self.train_df.index)
        self.test_df = self.valid_df.copy() # no need to use test set as we don't have access to labels for nih
        self.soln_df = self.valid_df.copy()
        self.diseases = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14']
        self.batch_size = args.batch_size

    def setup_data_mode(self):
        if self.data_mode == 'all':
            pass
        elif self.data_mode == 'frontal':
            self.train_valid_df = self.train_valid_df[self.train_valid_df["Path"].str.contains('frontal')]
            self.test_df = self.test_df[self.test_df["Path"].str.contains('frontal')]
        elif self.data_mode == 'lateral':
            self.train_valid_df = self.train_valid_df[self.train_valid_df["Path"].str.contains('lateral')]
            self.test_df = self.test_df[self.test_df["Path"].str.contains('lateral')]
        else:
            assert False, f"Unknown data mode: {self.data_mode}"

    def fill_dataset_nan(self, with_val, modify_validation=True):
        self.train_df.fillna(with_val, inplace=True)
        if modify_validation:
            self.valid_df.fillna(with_val, inplace=True)
    
    def remove_nan_rows(self):
        assert self.diseases != 'all', "Can't remove nan rows if using all diseases"
        self.train_df = self.train_df[~self.train_df[self.diseases[0]].isna()]
        self.valid_df = self.valid_df[~self.valid_df[self.diseases[0]].isna()]

    def fill_dataset_nan_with_mean(self, modify_validation=True):
        means = self.train_df[self.diseases].mean()
        self.train_df[self.diseases] = self.train_df[self.diseases].fillna(means)
        if modify_validation:
            self.valid_df[self.diseases] = self.valid_df[self.diseases].fillna(means)
    
    def fill_dataset_nan_with_random(self, min_val, max_val, modify_validation=True):
        self.train_df[self.diseases] = self.train_df[self.diseases].applymap(lambda x: np.random.uniform(min_val, max_val) if np.isnan(x) else x)
        if modify_validation:
            self.valid_df[self.diseases] = self.valid_df[self.diseases].applymap(lambda x: np.random.uniform(min_val, max_val) if np.isnan(x) else x)

    def shift_train_dataset_to_0_1_range(self):
        """
        Shifts the training dataset from [-1, 1] to [0, 1] range for disease columns
        Useful for bce loss
        """
        # self.train_df.fillna(fillna_with, inplace=True)
        self.train_df[self.diseases] = self.train_df[self.diseases].transform(lambda x: (x + 1) / 2)

    def shift_values_to_normal_range(self, vals):
        """
        Shifts the df from [0, 1] to [-1, 1] range for disease columns
        """
        return (vals - 0.5) * 2

    def save_model(self, model, model_name):
        model.save(f"{self.models_dir}/{model_name}.h5")
        print(f"Model saved to {self.models_dir}/{model_name}.h5")

    def load_model(self, model_name):
        return tf.keras.models.load_model(f"{self.models_dir}/{model_name}.h5")

    def save_preds(self, preds, output_file_name, mode, ensemble=False):
        if mode == "valid":
            df = self.valid_df
        elif mode == "test":
            df = self.test_df
        elif mode == "soln":
            df = self.soln_df
        else:
            assert False, f"Unknown mode: {mode}"
        preds_df = pd.DataFrame(preds)
        preds_df.columns = self.diseases
        ic(len(preds_df))
        preds_df.insert(0, "Id", df["Id"].values)
        if mode == "test":
            save_results_path = f"{self.results_dir}/{output_file_name}_TEST.csv"
        elif mode == "valid":
            save_results_path = f"{self.results_dir}/{output_file_name}_VALID.csv"
        elif mode == "soln":
            save_results_path = f"{self.results_dir}/{output_file_name}_SOLN.csv"
        if ensemble:
            save_results_path = save_results_path.replace(".csv", "_ENSEMBLE.csv")
        preds_df.to_csv(save_results_path, index=False)
        print(f"Results saved to {save_results_path}")

    def save_config_args(self, output_file_name, args):
        save_config_path = f"{self.results_dir}/{output_file_name}.json"
        with open(save_config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Config saved to {save_config_path}")

    def get_image_generators(self, img_target_size, augmentation_args, valid_test_image_generator_args = {}):
        augmentation_image_generator = ImageDataGenerator(**augmentation_args)
        valid_test_image_generator = ImageDataGenerator(**valid_test_image_generator_args)

        a = augmentation_image_generator.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.data_dir,
            x_col="Path",
            y_col=self.diseases,
            class_mode="raw",
            target_size=img_target_size,
            batch_size=self.batch_size,
            shuffle=True
        )
        b = valid_test_image_generator.flow_from_dataframe(
            dataframe=self.valid_df,
            directory=self.data_dir,
            x_col="Path",
            y_col=self.diseases,
            class_mode="raw",
            target_size=img_target_size,
            batch_size=self.batch_size,
            shuffle=True
        )
        c = valid_test_image_generator.flow_from_dataframe(
            dataframe=self.test_df,
            directory=self.data_dir,
            x_col="Path",
            shuffle=False,
            target_size=img_target_size,
            class_mode=None,
        )
        d = valid_test_image_generator.flow_from_dataframe(
            dataframe=self.soln_df,
            directory=self.data_dir,
            x_col="Path",
            shuffle=False,
            target_size=img_target_size,
            class_mode=None,
        )
        return a, b, c, d