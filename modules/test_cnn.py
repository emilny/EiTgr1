"""
This module is for training and testing different kinds of Convolutional neural networks
Perform comparisons and choose the network configuration that yields the best results!
"""
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
import preprocessing
from modules.transferlearning import gennet_transfer_learning
from modules.baseline import gennet_baseline


def test_accuracy(model, test_x, test_y):
    """
    Function for testing a finished model on the final test set
    :param model: Trained model
    :param test_x: featureset
    :param test_y: labels for featuresets
    :return: binary accuracy score as Correct/Correct+False
    """
    predictions = model.predict(test_x)
    i = 0
    sum = 0
    for p in predictions:
        diff = np.argmax(p) - np.argmax(test_y[i])
        if diff == 0:
            sum += 1
        i += 1
    return sum/i


def get_generators(percentage=None):
    """
    Create generators for data augmentation models, if percentage is not given, the data is assumed to be
    preprocessed already
    :param percentage: Percent of total data set to include ( <100% for preliminary testing)
    :return: Two data generators and the test set
    """

    if percentage is not None:
        preprocessing.create_dataset(percentage, training=True, augmented=True)
        preprocessing.create_dataset(percentage, training=False)
    train_datagen, val_datagen = preprocessing.create_train_and_validation_gens(batch_sizes=(30, 20), validation_split=0.2)
    X_test, Y_test = preprocessing.load_dataset(train=False)
    return train_datagen, val_datagen, X_test, Y_test

def prep_train_data(percentage=None):
    """
    Create numpy arrays from data sets.
    NOTE: This has to be done differently for data augmentation model, use get_generators instead
    :param percentage: Percent of total data set to include ( <100% for preliminary testing)
    :return: Train and test set (train is not split into train/validation yet)
    """

    if percentage is not None:
        preprocessing.create_dataset(percentage, training=True, augmented=False)
        preprocessing.create_dataset(percentage, training=False)
    X_train, Y_train = preprocessing.load_dataset(train=True)  # Load train data, np.arrays
    X_test, Y_test = preprocessing.load_dataset(train=False)  # Load test data, np.arrays
    return X_train, Y_train, X_test, Y_test


def train_test_model(name, model, X_train, Y_train, X_test, Y_test, validation_split=0.1):
    """
    Train a model that has already been compiled, on the given data set, and then test it on the test set.
    :param model: A precompiled model, untrained (or pre-trained as VGG16 etc)
    :param X_train: Train featuresets
    :param Y_train: Train labels
    :param X_test: Test featuresets
    :param Y_test: Test labels
    :param validation_split: Percent of train set to hold out for validation, defaults to 10%
    :return:
    """

    # Create tensorboard callback for visualisations of training process
    tensorboard_callback = TensorBoard(log_dir=f"./logs_{name}")

    # Create checkpointer to save best model at each epoch, making sure we capture the best before overfitting occurs
    checkpointer = ModelCheckpoint(filepath=f"../models/best_{name}.hdf5", save_best_only=True)

    # Train model using ordinary fit
    model.fit(x=X_train, y=Y_train, batch_size=30, epochs=50, validation_split=validation_split,
              callbacks=[tensorboard_callback, checkpointer])

    model.load_weights(f"best_{name}.hdf5")
    accuracy = test_accuracy(model, X_test, Y_test)

    print(f"Accuracy on test set for the baseline model was: {accuracy*100}%")
    # Does not save model at this point, this should be implemented along with fit_generator and checkpointer


def train_test_model_data_augmentation(name, model, train_datagen, val_datagen, X_test, Y_test):
    """
    This method uses slightly different input approach since data is augmented on the fly,
    i.e. we need to pass data-generators instead of x_train (and a separate validation datagen, which only rescales imgs)

    :param train_datagen: Data augmentation generator, takes in X_data and outputs augmented pictures
    :param val_datagen: Just a rescaling generator for validation samples
    :param X_test:
    :param Y_test:
    :return: None
    """

    # Create tensorboard callback for visualisations of training process
    tensorboard_callback = TensorBoard(log_dir=f"./logs_{name}")

    # Create checkpointer to save best model at each epoch, making sure we capture the best before overfitting occurs
    checkpointer = ModelCheckpoint(filepath=f"../models/best_{name}.hdf5", save_best_only=True)

    # Train model using  fit_generator
    epochs = 50
    steps_per_epoch = len(train_datagen)
    model.fit(train_datagen,
              validation_data=val_datagen,
              steps_per_epoch=steps_per_epoch,  # Batch size is 30 from generator
              #batch_size=100,
              epochs=epochs,
              validation_steps=len(val_datagen),
              callbacks=[tensorboard_callback, checkpointer])

    model.load_weights(f"best_{name}.hdf5")
    accuracy = test_accuracy(model, X_test, Y_test)

    print(f"Accuracy on test set for model on augmented data was: {accuracy}")
    # Does not save model at this point, this should be implemented along with fit_generator and checkpointer





if __name__ == '__main__':

    # TODO Test the two different kinds of models with and without both focal loss and data augmentation:
    # Transfer learning
    # Baseline
    # TODO Find out about suitable parameters and make an educated guess

    x_shape = (100, 100, 3)  # Hardcoded for now

    model_baseline = gennet_baseline(x_shape=x_shape)
    #model_transfer_learning = gennet_transfer_learning(x_shape)

    t_datagen, val_datagen, X_test, Y_test = get_generators(percentage=None)
    preprocessing.create_dataset(1, training=False)
    X_test, Y_test = preprocessing.load_dataset(train=False)

    model_baseline.load_weights(f"best_first_baseline_test.hdf5")
    accuracy = test_accuracy(model_baseline, X_test, Y_test)
    print(accuracy)
    # X_train, Y_train, X_test, Y_test = prep_train_data(percentage=None)
    #train_test_model_data_augmentation("first_baseline_test",
    #                                   model_baseline,
    #                                   t_datagen,
    #                                   val_datagen,
    #                                   X_test,
    #                                  Y_test)
    # train_test_model(model_baseline,X_train,Y_train, X_test,Y_test)


# Test accuracy for hele datasettet:
# 0.826 for focal med gamma = 2
# 0.821 for crossentropy
# 0.821 for focal loss med gamma = 1
# 0.814 for focal loss med gamma = 3
# 0.854 for baseline med data augmentation
