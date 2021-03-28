import numpy as np  # Linalg library
from keras.callbacks import TensorBoard, ModelCheckpoint
import preprocessing
from modules.transferlearning import gennet_transfer_learning
from modules.baseline import gennet_baseline
from focal_loss import focal_loss


def test_accuracy(model, test_x, test_y):
    """
    Function for testing a finished model on the final test set
    :param model: Fully trained model
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


def prep_generators(percentage=None):
    # Create train and validation data sets and store these separately. Split defaults to 10%
    if percentage is not None:
        preprocessing.create_dataset(percentage, training=True, augmented=True)
        preprocessing.create_dataset(percentage, training=False)
    train_datagen, val_datagen = preprocessing.create_train_and_validation_gens(1, validation_split=0.1)
    X_test, Y_test = preprocessing.load_dataset(train=False)
    return train_datagen, val_datagen, X_test, Y_test

def prep_train_data(percentage=None):
     # Create numpy arrays from data sets. This has to be done differently for data augmentation model
    if percentage is not None:
        preprocessing.create_dataset(percentage, training=True, augmented=False)
        preprocessing.create_dataset(percentage, training=False)
    X_train, Y_train = preprocessing.load_dataset(train=True) # Load train data, np.arrays
    X_test, Y_test = preprocessing.load_dataset(train=False) # Load test data, np.arrays
    return X_train, Y_train, X_test, Y_test


def train_test_model(model, X_train, Y_train, X_test, Y_test, validation_split=0.1):
    # Create tensorboard callback for visualisations of training process
    tensorboard_callback = TensorBoard(log_dir="./logs")


    # Train model using ordinary fit TODO: Update this to fit_generator to capture model at best before overfitting
    model.fit(x = X_train, y = Y_train, batch_size=100, epochs=20, validation_split=validation_split,
              callbacks=[tensorboard_callback])

    accuracy = test_accuracy(model, X_test, Y_test)

    print(f"Accuracy on test set for the baseline model was: {accuracy*100}%")
    # Does not save model at this point, this should be implemented along with fit_generator and checkpointer


def train_test_model_data_augmentation(model, train_datagen, val_datagen, X_test, Y_test):
    """
    This method uses slightly different input approach since data is augmented on the fly,
    i.e. we need to pass data-generators instead of x_train (and a separate validation datagen, which only rescales)

    :param train_datagen: Data augmentation generator, takes in X_data and outputs augmented pictures
    :param val_datagen: Just a rescaling generator for validation samples
    :param X_test:
    :param Y_test:
    :return: None
    """


    # Create tensorboard callback for visualisations of training process
    tensorboard_callback = TensorBoard(log_dir="../logs")

    # Train model using ordinary fit TODO: Update this to fit_generator to capture model at best before overfitting
    model.fit(train_datagen,
              validation_data=val_datagen,
              batch_size=100,
              epochs=20,
              callbacks=[tensorboard_callback])

    accuracy = test_accuracy(model, X_test, Y_test)

    print(f"Accuracy on test set for model on augmented data was: {accuracy}")
    # Does not save model at this point, this should be implemented along with fit_generator and checkpointer





if __name__ == '__main__':
    x_shape = (100, 100, 3) # Hardcoded for now

    #model_baseline = gennet_baseline(name="Test01", x_shape=x_shape)
    model_transfer_learning = gennet_transfer_learning(x_shape)

    t_datagen, val_datagen, X_test, Y_test = prep_generators(percentage=None)
    train_test_model_data_augmentation(model_transfer_learning, t_datagen, val_datagen, X_test, Y_test)
    #train_test_model(model_baseline,X_train,Y_train, X_test,Y_test)








"""    

#preprocessing.create_dataset(1, training=False)

MODEL_NAME = 'covid_test-{}-{}.model'.format(LR, '2conv-basic')

#model = load_model(filepath=f"models/{MODEL_NAME}")





#X, Y = X[0:int(len(X)*0.1)], Y[0:int(len(X)*0.1)]

x_shape = X[0].shape
input_tens = Input(shape=x_shape)

#model = Sequential()
#pre_trained = vgg16.VGG16(include_top=False, input_tensor=input_tens, pooling="max")




"""
"""
# Create first layer (to receive input)
model.add(layer=Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=x_shape))
model.add(layer=MaxPool2D(pool_size=(2, 2)))

# Create additional Convolutional layers
filters = [32, 64]
for f in filters:
    # Adding several conv layers with different filter sizes
    model.add(layer=Conv2D(filters=f, kernel_size=(3, 3), activation="relu"))
    model.add(layer=MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)
"""
"""


#model.add(layer=Flatten())
#model.add(layer=Dense(units=1024, activation="relu"))
#model.add(layer=Dense(units=1024, activation="relu"))
#model.add(layer=Dense(units=3, activation="softmax"))  # Output is a 3-vector

model.compile(optimizer=optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=['binary_accuracy',
                       metrics.FalsePositives(),
                       metrics.FalseNegatives(),
                       metrics.TruePositives(),
                       metrics.TrueNegatives()])

if os.path.exists(f"models/{MODEL_NAME}"):
    pass
    # model.load_weights(filepath=f"models/{MODEL_NAME}")
    # print('model loaded!')

#tensorboard_callback = TensorBoard(log_dir="./logs")

#checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint',
#                               mode='max', verbose=2, save_best_only=True)

#model.fit(x = X, y = Y, batch_size=100, epochs=20, validation_split=0.1,
#          callbacks=[tensorboard_callback])

history = model.fit_generator(train_datagen, #steps_per_epoch=len(X)//20,
                    validation_data=val_datagen,
                    #validation_steps=len(X_val)//20,
                    epochs=20,
                    verbose=2)

#model.save_weights(filepath=f"models/{MODEL_NAME}")

test_x, test_y = preprocessing.load_dataset(train=False)


# testing:



test_accuracy(model, test_x)

# Test accuracy for hele datasettet:
# 0.826 for focal med gamma = 2
# 0.821 for crossentropy
# 0.821 for focal loss med gamma = 1
# 0.814 for focal loss med gamma = 3
"""
