import numpy as np  # Linalg library
import pandas as pd  # Data processing, easy to handle dataframes
import cv2  # Computer vision library, image handling
import matplotlib.pyplot as plt  # For checking images along the way
import random  # For shuffling data
import os
from tqdm import tqdm
from tensorflow import keras

TRAINING_DATA_DIR = "Data/train/"
TEST_DATA_DIR = "Data/test/"

CATEGORIES = ["COVID19", "NORMAL", "PNEUMONIA"]

cats_to_onehots = {"COVID19": [1, 0, 0],
                   "NORMAL": [0, 1, 0],
                   "PNEUMONIA": [0, 0, 1]}

IMG_SIZE = 100

CLASS_WEIGHTS = {0: 50.0,  # React more heavily to COVID-19 cases, since set is unbalanced
                 1: 25.0,
                 2: 25.0}


def create_dataset(percentage_of_data_set, training=True):
    """
    Creates dataset as features, labels from data directory
    and saves to numpy files to save space
    :param percentage_of_data_set: how much of the total dataset to process
    :param training: type of data set to create
    :return: None
    """
    data_dir = TRAINING_DATA_DIR if training else TEST_DATA_DIR
    data_x = []
    data_y = []
    for cat in CATEGORIES:
        onehot = np.array(cats_to_onehots[cat])
        path = os.path.join(data_dir, cat)
        for i, img in enumerate(tqdm(os.listdir(path), colour="#39FF14")):
            if i > percentage_of_data_set * len(os.listdir(path)):
                break
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data_x.append(np.array(resized_array))
            data_y.append(onehot)

    index_list = [i for i in range(len(data_x))]
    np.random.shuffle(index_list)
    x_data = [data_x[i] for i in index_list]
    y_data = [data_y[i] for i in index_list]
    if training:
        np.save('training_data', x_data)
        np.save('training_labels', y_data)
    else:
        np.save('test_data', x_data)
        np.save('test_labels', y_data)


def load_dataset(train=True):
    """
    Loads train or test data from previously saved numpy arrays
    :param train: determines data set
    :return: features, labels
    """
    type = "training" if train else "test"
    filename_x = f"{type}_data.npy"
    filename_y = f"{type}_labels.npy"
    x_data = np.load(filename_x)
    y_data = np.load(filename_y)
    norm_x_data = keras.utils.normalize(x_data)
    return norm_x_data, y_data
