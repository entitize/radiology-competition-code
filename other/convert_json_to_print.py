# helper code to convert the competition json to the format for the google sheets
import re
input_str = input("Enter the json string: ")

diseases = ["Atelectasis", "Average MSE", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
json_str = eval(input_str)
for disease in diseases:
  print(json_str[disease])



