from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers, metrics
from keras.callbacks import TensorBoard

import preprocessing
import os

#preprocessing.create_dataset(0.5, training=True)
#preprocessing.create_dataset(1, training=False)

LR = 0.01
MODEL_NAME = 'covid_test-{}-{}.model'.format(LR, '2conv-basic')

#model = load_model(filepath=f"models/{MODEL_NAME}")

X, Y = preprocessing.load_dataset(train=True)

x_shape = X[0].shape

model = Sequential()

# Create first layer (to receive input)
model.add(layer=Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=x_shape))
model.add(layer=MaxPool2D(pool_size=(2, 2)))

# Create additional Convolutional layers
filters = [32, 64]
for f in filters:
    # Adding several conv layers with different filter sizes
    model.add(layer=Conv2D(filters=f, kernel_size=(3, 3), activation="relu"))
    model.add(layer=MaxPool2D(pool_size=(2, 2)))

model.add(layer=Flatten())
model.add(layer=Dense(units=1024, activation="relu"))
model.add(layer=Dense(units=3, activation="softmax"))  # Output is a 3-vector

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

tensorboard_callback = TensorBoard(log_dir="./logs")
model.fit(x=X,
          y=Y,
          batch_size=100,
          epochs=100,
          validation_split=0.1,
          verbose=0,
          callbacks=[tensorboard_callback])

model.save_weights(filepath=f"models/{MODEL_NAME}")



