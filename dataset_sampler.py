# Extracts random samples from the dataset and saves them in a new folder
import os
import shutil
import random

# set random seed
random.seed(42)
# Folder names of classes to be sampled
folder_names = ["call", "rock", "dislike", "fist", "palm", "peace", "one", "ok", "like"]
# Directory path for the dataset
dataset_dir = "./hagrid-classification-128p-squares/"
# Number of samples to extract
number_of_samples = 100

for folder in folder_names:
    file_names = os.listdir(f'{dataset_dir}/{folder}')
    random_files = random.sample(file_names, number_of_samples)
    os.makedirs(f'./samples/{folder}', exist_ok=True)
    i=1
    [shutil.copy2(f'./{dataset_dir}/{folder}/{file}', f'./samples/{folder}/{i+1}{file[-5:]}') for i, file in enumerate(random_files)]