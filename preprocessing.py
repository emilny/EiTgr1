import numpy as np  # Linalg library
import pandas as pd  # Data processing, easy to handle dataframes
import cv2  # Computer vision library, image handling
import matplotlib.pyplot as plt  # For checking images along the way
import random  # For shuffling data
import os
from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

TRAINING_DATA_DIR = "/Users/emilny/Downloads/Data/train/"
TEST_DATA_DIR = "/Users/emilny/Downloads/Data/test/"

CATEGORIES = ["COVID19", "NORMAL", "PNEUMONIA"]

cats_to_onehots = {"COVID19": [1.0, 0.0, 0.0],
                   "NORMAL": [0.0, 1.0, 0.0],
                   "PNEUMONIA": [0.0, 0.0, 1.0]}

IMG_SIZE = 100

CLASS_WEIGHTS = {0: 50.0,  # React more heavily to COVID-19 cases, since set is unbalanced
                 1: 25.0,
                 2: 25.0}


def create_dataset(percentage_of_data_set=1., training=True, augmented=False):
    """
    Creates dataset as features, labels from data directory
    and saves to numpy files to save space
    :param augmented: Determines whether we are prepping data for augmentation generators or normalizing
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
        for i, img in enumerate(tqdm(os.listdir(path), colour='#39ff14')):
            if i > percentage_of_data_set * len(os.listdir(path)):
                break
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data_x.append(np.array(resized_array))
            data_y.append(onehot)
    data_x = np.array(data_x)

    index_list = [i for i in range(len(data_x))]
    np.random.shuffle(index_list)
    x_data = [data_x[i] for i in index_list]
    y_data = [data_y[i] for i in index_list]

    if not augmented:  # Don't want to normalize when we are using Datagenerators
        x_data = keras.utils.normalize(x_data)

    if training:
        print("Training data saved!")
        np.save('training_data', x_data)
        np.save('training_labels', y_data)
    else:
        print("Test data saved!")
        np.save('test_data', x_data)
        np.save('test_labels', y_data)


def create_train_and_validation_gens(batch_size, validation_split):
    """
    Augments images by performing several (=batch_size) augmenting transformations
    Saves augmented images to augmented folder
    :param validation_split: Split percentage for validation set
    :return: Two data generators, one for train and one for validation, validation is not augmented
    """
    train_x, train_y = load_dataset(train=True)  # Load full train set

    # Perform validation split
    split_idx = int(len(train_x) * validation_split)

    train_x, train_y, val_x, val_y = train_x[split_idx:], train_y[split_idx:], train_x[:split_idx], train_y[:split_idx]

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       zoom_range=0.3,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen = train_datagen.flow(train_x, train_y, batch_size=batch_size)
    validation_datagen = validation_datagen.flow(val_x, val_y, batch_size=batch_size)

    return train_datagen, validation_datagen


def load_dataset(train=True):
    """
    Loads train or test data from previously saved numpy arrays
    :param train: determines data set
    :return: features, labels
    """

    typ = "training" if train else "test"
    filename_x = f"{typ}_data.npy"
    filename_y = f"{typ}_labels.npy"
    x_data = np.load(filename_x)
    y_data = np.load(filename_y)
    return x_data, y_data


if __name__ == '__main__':
    create_dataset(percentage_of_data_set=0.1, training=True, augmented=True)
    train_datagen, val_datagen = create_train_and_validation_gens(1, validation_split=0.1)

    imgtrain = [next(train_datagen) for i in range(0, 5)]
    imgval = [next(val_datagen) for i in range(0, 5)]

    fig, ax = plt.subplots(1, 10, figsize=(16, 6))
    # print('Labels:', [item[1][0] for item in img])
    for i in range(0, 10):
        if i < 5:
            ax[i].imshow(imgval[i][0][0])
        else:
            ax[i].imshow(imgtrain[i - 5][0][0])

    plt.show()
