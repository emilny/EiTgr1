import numpy as np  # Linalg library
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers, metrics
from keras.callbacks import TensorBoard

import preprocessing
import os
#from preprocessing import datagen



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

model.fit(x = X, y = Y, batch_size=100, epochs=20, validation_split=0.1,
          verbose = 2, callbacks=[tensorboard_callback])

# model.fit_generator(datagen.flow(X, Y, batch_size=100),
#           epochs=30,
#           steps_per_epoch=len(X)//100,
#           verbose=1,
#           callbacks=[tensorboard_callback])

model.save_weights(filepath=f"models/{MODEL_NAME}")

test_x, test_y = preprocessing.load_dataset(train=False)

predictions = model.predict(test_x)

# testing:

i = 0
sum = 0
for p in predictions:
    diff = np.argmax(p) - np.argmax(test_y[i])
    if diff == 0:
        sum += 1
    i += 1
print(sum/i)

