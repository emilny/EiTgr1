import numpy as np  # Linalg library
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from keras.models import Sequential, load_model, Model
from keras import optimizers, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications import vgg16

import preprocessing
import os
#from preprocessing import datagen
from focal_loss import focal_loss





preprocessing.create_dataset(0.1, training=True, augmented = True)
preprocessing.create_dataset(0.1, training=False)
#preprocessing.create_dataset(1, training=False)

LR = 0.01
MODEL_NAME = 'covid_test-{}-{}.model'.format(LR, '2conv-basic')

#model = load_model(filepath=f"models/{MODEL_NAME}")


X, Y = preprocessing.load_dataset(train=True)
X_val, Y_val = preprocessing.load_dataset(train = True, validation=True)
train_datagen, val_datagen = preprocessing.create_train_and_validation_gens(2)


#X, Y = X[0:int(len(X)*0.1)], Y[0:int(len(X)*0.1)]

x_shape = X[0].shape
input_tens = Input(shape=x_shape)

#model = Sequential()
#pre_trained = vgg16.VGG16(include_top=False, input_tensor=input_tens, pooling="max")



# load model without classifier layers
pre_trained = vgg16.VGG16(include_top=False, input_shape=x_shape)
for layer in pre_trained.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(pre_trained.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(3, activation='softmax')(class1)
# define new model
model = Model(inputs=pre_trained.inputs, outputs=output)
# summarize
#model.summary()

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

model.fit_generator(train_datagen, steps_per_epoch=len(X)//20,
                    validation_data=val_datagen,
                    validation_steps=len(X_val)//20,
                    epochs=20,
                    verbose=2)

model.save_weights(filepath=f"models/{MODEL_NAME}")

test_x, test_y = preprocessing.load_dataset(train=False)


# testing:

def test_accuracy(model, test_x):
    predictions = model.predict(test_x)
    i = 0
    sum = 0
    for p in predictions:
        diff = np.argmax(p) - np.argmax(test_y[i])
        if diff == 0:
            sum += 1
        i += 1
    print(sum/i)

test_accuracy(model, test_x)

# Test accuracy for hele datasettet:
# 0.826 for focal med gamma = 2
# 0.821 for crossentropy
# 0.821 for focal loss med gamma = 1
# 0.814 for focal loss med gamma = 3
