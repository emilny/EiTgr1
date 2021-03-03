# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:01:13 2021

@author: 47924
"""
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers, metrics
from keras.callbacks import TensorBoard
from keras.applications import Xception
import preprocessing
import os

X, Y = preprocessing.load_dataset(train=True)
x_shape = X[0].shape

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(x_shape),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False #freeze the base model.

inputs = keras.Input(shape=(x_shape))

x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(3)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

tensorboard_callback = TensorBoard(log_dir="./logs")

model.fit(x=X,
          y=Y, 
          batch_size=100,
          epochs=20,
          validation_split=0.1,
          verbose=0,
          callbacks=[tensorboard_callback])

