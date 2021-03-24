import numpy as np  # Linalg library
import pandas as pd  # Data processing, easy to handle dataframes
import cv2  # Computer vision library, image handling
import matplotlib.pyplot as plt  # For checking images along the way
import random  # For shuffling data
import os
from tqdm import tqdm
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator



TRAINING_DATA_DIR = r"C:\Users\eivol\OneDrive\Dokumenter\GitHub/Data/train/"
TEST_DATA_DIR = r"C:\Users\eivol\OneDrive\Dokumenter\GitHub\Data/test/"

CATEGORIES = ["COVID19", "NORMAL", "PNEUMONIA"]

cats_to_onehots = {"COVID19": [1.0, 0.0, 0.0],
                   "NORMAL": [0.0, 1.0, 0.0],
                   "PNEUMONIA": [0.0, 0.0, 1.0]}

IMG_SIZE = 100

CLASS_WEIGHTS = {0: 50.0,  # React more heavily to COVID-19 cases, since set is unbalanced
                 1: 25.0,
                 2: 25.0}


def create_dataset(percentage_of_data_set=1., training=True, augmented = False, validation_split = 0.1):
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
        for i, img in enumerate(tqdm(os.listdir(path))):
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


    if training and augmented:
        split_index = int(len(x_data)*validation_split)
        train_data,  validation_data = x_data[split_index:], x_data[:split_index]
        train_labels, validation_labels = y_data[split_index:], y_data[:split_index]
        y_data = train_labels
        norm_x_data = train_data # dont want to normalize when using ImgGenerator
        np.save("validation_data", validation_data)
        np.save("validation_labels", validation_labels)
    else:
        norm_x_data = keras.utils.normalize(x_data)

    if training:
        print("Saved!")
        np.save('training_data', norm_x_data)
        np.save('training_labels', y_data)
    else:
        np.save('test_data', norm_x_data)
        np.save('test_labels', y_data)

    # This is where training data is increased in size by way of data augmentation:

def create_train_and_validation_gens(batch_size):
    """
    Augments images by performing several (=batch_size) augmenting transformations
    Saves augmented images to augmented folder
    :param orig_x: Input images as np.arrays
    :return:
    """
    train_x, train_y = load_dataset(train = True)
    val_x, val_y = load_dataset(train = True, validation=True)

    train_datagen = ImageDataGenerator(rescale=1./255,
                                 zoom_range=0.3,
                                 rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                       zoom_range=0.3,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')



    train_datagen = train_datagen.flow(train_x, train_y, batch_size = batch_size)
    validation_datagen = validation_datagen.flow(val_x, val_y, batch_size=batch_size)


    #img = [next(datagen) for i in range(0, 5)]

    #fig, ax = plt.subplots(1, 5, figsize=(16, 6))
    #print('Labels:', [item[1][0] for item in img])
    #for i in range(0, 5):
    #    ax[i].imshow(img[i][0][0])
    #plt.show()

    return train_datagen, validation_datagen


def load_dataset(train=True, validation = False):
    """
    Loads train or test data from previously saved numpy arrays
    :param train: determines data set
    :return: features, labels
    """
    if validation:
        typ = "validation"
    else:
        typ = "training" if train else "test"
    filename_x = f"{typ}_data.npy"
    filename_y = f"{typ}_labels.npy"
    x_data = np.load(filename_x)
    y_data = np.load(filename_y)


    return x_data, y_data

