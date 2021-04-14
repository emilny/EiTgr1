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
    sum = 0
    for i, p in enumerate(predictions):
        diff = np.argmax(p) - np.argmax(test_y[i])
        if diff == 0:
            sum += 1
    return sum/len(predictions)

def test_false_P_N(model, test_x, test_y):
    """
    Function for testing a finished model on the final test set,
    focus on False Covid predictions, either falsely positive or falsely negative
    :param model: Trained model
    :param test_x: featureset
    :param test_y: labels for featuresets
    :return: False positives, False negatives as percentage for Covid cases: FPR = FP/FP+TN, FNR= FN/FN+TP
    """

    predictions = model.predict(test_x)
    fp = 0
    fn = 0
    tn = 0
    tp = 0
    for i, p in enumerate(predictions):
        covid_predicted = np.argmax(p) == 0
        actually_covid = np.argmax(test_y[i]) == 0
        if covid_predicted:  # Positive
            if actually_covid:  # True
                tp += 1
            else:  # False
                fp += 1
        else:  # Negative
            if actually_covid:  # False
                fn += 1
            else:  # True:
                tn += 1
    assert fp + fn + tp + tn == len(predictions)

    # Avoid zero-division in next segment
    if fp + tn == 0:
        tn = 1
    if fn + tp == 0:
        tp = 1

    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)

    return fpr, fnr


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
    checkpointer = ModelCheckpoint(filepath=f"./models/best_{name}.hdf5", save_best_only=True)

    # Train model using ordinary fit
    model.fit(x=X_train, y=Y_train, batch_size=30, epochs=50, validation_split=validation_split,
              callbacks=[tensorboard_callback, checkpointer])

    model.load_weights(f"./models/best_{name}.hdf5")
    accuracy = test_accuracy(model, X_test, Y_test)
    fpr, fnr = test_false_P_N(model, X_test, Y_test)

    print(f"Accuracy on test set for model on augmented data was: {accuracy*100}%")
    print(f"(COVID) False positive rate on test set for model on augmented data was: {fpr*100}%")
    print(f"(COVID) False negative rate on test set for model on augmented data was: {fnr*100}%")



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
    checkpointer = ModelCheckpoint(filepath=f"./models/best_{name}.hdf5", save_best_only=True)

    # Train model using  fit_generator
    epochs = 30
    steps_per_epoch = len(train_datagen)
    model.fit(train_datagen,
              validation_data=val_datagen,
              steps_per_epoch=steps_per_epoch,  # Batch size is 30 from generator
              #batch_size=100,
              epochs=epochs,
              validation_steps=len(val_datagen),
              callbacks=[tensorboard_callback, checkpointer])

    model.load_weights(f"./models/best_{name}.hdf5")
    accuracy = test_accuracy(model, X_test, Y_test)
    fpr, fnr = test_false_P_N(model, X_test, Y_test)

    print(f"Accuracy on test set for model on augmented data was: {accuracy*100}%")
    print(f"(COVID) False positive rate on test set for model on augmented data was: {fpr*100}%")
    print(f"(COVID) False negative rate on test set for model on augmented data was: {fnr*100}%")





if __name__ == '__main__':

    # TODO Test the two different kinds of models with and without both focal loss and data augmentation:
    # Transfer learning
    # TODO: Finn ut om 1: flere dense på bunnen er bra? 2: Færre trainable=False på slutten?
    #   0.XXX med/uten focal med/uten data augmentation
    # Baseline (Satt med parametere fra artikkel)
    #   0.XXX med/uten focal med/uten data augmentation

    x_shape = (100, 100, 3)  # Hardcoded for now (Argument: Større tar fette lang tid å kjøre)

    model_baseline = gennet_baseline(x_shape=x_shape, use_focal=True)
    #model_transfer_learning = gennet_transfer_learning(x_shape)

    #preprocessing.create_dataset(1, training=False, augmented=True)
    t_datagen, val_datagen, X_test, Y_test = get_generators(percentage=None)  # FOR DATA AUGMENTATION
    # X_train, Y_train, X_test, Y_test = prep_train_data(percentage=None)  # FOR NORMAL DATA TRAINING

    train_test_model_data_augmentation("baseline_augment_focal_loss",
                                       model_baseline,
                                       t_datagen,
                                       val_datagen,
                                       X_test,
                                       Y_test)
    # train_test_model(model_baseline,X_train,Y_train, X_test,Y_test)


# Test accuracy for hele datasettet:
# 0.826 for focal med gamma = 2
# 0.821 for crossentropy
# 0.821 for focal loss med gamma = 1
# 0.814 for focal loss med gamma = 3
# 0.854 for baseline med data augmentation
