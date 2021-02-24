from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn import DNN
import preprocessing
import os

#preprocessing.create_dataset(1)

LR = 0.01
MODEL_NAME = 'covid_test-{}-{}.model'.format(LR, '2conv-basic')

convnet = input_data(shape=[64, preprocessing.IMG_SIZE, preprocessing.IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

X, Y = preprocessing.load_dataset(train=True)
model.fit(X_inputs=X,
          Y_targets=Y,
          n_epoch=3,
          validation_batch_size=0.1,
          snapshot_step=500,
          show_metric=True,
          run_id=MODEL_NAME)

