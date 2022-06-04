""" Picker program. For each column, choose which file to use for the final csv. """
import pandas as pd
import argparse
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, required=True, help='Output file name.')
args = parser.parse_args()

diseases = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

final_df = pd.DataFrame(columns=diseases)

for disease in diseases:
    # ask for the file name
    df = None
    while True:
        logger.info("Picking for " + disease)
        file_name = input("Enter the file name (omit ../results/): ")
            # check if file exists
        try:
            df = pd.read_csv("../results/" + file_name)
            break
        except FileNotFoundError:
            logger.error("File not found. Please try again.")
            continue
    final_df[disease] = df[disease]
    final_df["Id"] = df["Id"]

results_dir = "../results/"
final_df.to_csv(results_dir + args.output_file, index=False)
logger.success(f"Picked results saved to ../results/{args.output_file}")
logger.info("Pro tip: hold command and click the file name above to open the file.")


    


