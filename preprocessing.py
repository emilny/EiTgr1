import numpy as np  # Linalg library
import pandas as pd  # Data processing, easy to handle dataframes
import cv2  # Computer vision library, image handling
import matplotlib.pyplot as plt  # For checking images along the way
import random  # For shuffling data
import os

TRAINING_DATA_DIR = "Data/train/"
TEST_DATA_DIR = "Data/test/"

CATEGORIES = ["COVID19", "NORMAL", "PNEUMONIA"]

cats_to_onehots = {"COVID19": [1, 0, 0],
                   "NORMAL": [0, 1, 0],
                   "PNEUMONIA": [0, 0, 1]}

IMG_SIZE = 100

def create_dataset(data_dir, percentage_of_data_set):
    data_x = []
    data_y = []
    for cat in CATEGORIES:
        onehot = np.array(cats_to_onehots[cat])
        path = os.path.join(data_dir, cat)
        for i, img in enumerate(os.listdir(path)):
            if i > percentage_of_data_set*len(os.listdir(path)):
                break
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data_x.append(np.array(resized_array))
            data_y.append(onehot)

    index_list = [i for i in range(len(data_x))]
    np.random.shuffle(index_list)
    x_data = [data_x[i] for i in index_list]
    y_data = [data_y[i] for i in index_list]
    return x_data, y_data

create_dataset(TRAINING_DATA_DIR, 0.01)  # Generates 1% of the training data set



