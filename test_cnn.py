from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers, metrics

import preprocessing
import os

# preprocessing.create_dataset(1)

LR = 0.01
MODEL_NAME = 'covid_test-{}-{}.model'.format(LR, '2conv-basic')

X, Y = preprocessing.load_dataset(train=True)

x_shape = X[0].shape

model = Sequential()

filters = [16, 32, 64]
for f in filters:
    # Adding several conv layers with different filter sizes
    model.add(layer=Conv2D(filters=f, kernel_size=(3, 3), activation="relu", input_shape=x_shape))
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

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

model.fit(x=X,
          y=Y,
          epochs=3,
          validation_split=0.1,
          verbose=1)
