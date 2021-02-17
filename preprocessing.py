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

def create_train_data():
    train_x = []
    train_y = []
    for cat in CATEGORIES:
        onehot = np.array(cats_to_onehots[cat])
        path = os.path.join(TRAINING_DATA_DIR, cat)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_x.append(np.array(resized_array))
            train_y.append(onehot)

    index_list = [i for i in range(len(train_x))]
    np.random.shuffle(index_list)
    x_train = [train_x[i] for i in range(len(train_x))]
    y_train = [train_y[i] for i in range(len(train_y))]

    
