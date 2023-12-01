# Extracts random samples from the dataset and saves them in a new folder
import os
import shutil
import random

# Folder names of classes to be sampled
folder_names = ['one', 'two_up', 'three', 'four', 'palm']
# Directory path for the dataset
dataset_dir = "./hagrid-classification-512p"
# Number of samples to extract
number_of_samples = 10

for folder in folder_names:
    file_names = os.listdir(f'{dataset_dir}/{folder}')
    random_files = random.sample(file_names, number_of_samples)
    os.makedirs(f'./samples/{folder}', exist_ok=True)
    i=1
    for file in random_files:
        shutil.copy2(f'./{dataset_dir}/{folder}/{file}', f'./samples/{folder}/{i}{file[-5:]}')
        i+=1