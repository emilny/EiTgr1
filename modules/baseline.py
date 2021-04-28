from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras import optimizers
from modules.focal_loss import focal_loss


def gennet_baseline(x_shape, use_focal=False):
    """
    Generates a network with some hard-coded parameters, compiles and returns an untrained model
    :param x_shape: Shape of input layer to accommodate input features
    :param use_focal: Whether or not to use focal loss as loss function. False: categorical crossentropy
    :return: untrained model
    """
    model = Sequential()

    # Create first layer (to receive input)
    model.add(layer=Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=x_shape))
    model.add(layer=MaxPool2D(pool_size=(2, 2)))

    # Create additional Convolutional layers
    filters = [64, 128, 256, 512]
    for f in filters:
        # Adding several conv layers with different filter sizes
        model.add(layer=Conv2D(filters=f, kernel_size=(3, 3), activation="relu"))
        model.add(layer=MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))  # TODO: Kanskje 50% er litt i overkant?
    model.add(layer=Flatten())
    model.add(layer=Dense(units=1024, activation="relu"))
    model.add(layer=Dense(units=1024, activation="relu"))
    model.add(layer=Dense(units=1024, activation="relu"))
    model.add(layer=Dense(units=3, activation="softmax"))  # Output is a 3-vector

    model.compile(optimizer=optimizers.Adam(),
                  loss=focal_loss(alpha=1.0, gamma=2.0) if use_focal else "categorical_crossentropy",
                  metrics=['binary_accuracy'])
    return model
