from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers, metrics
from keras.callbacks import TensorBoard
import preprocessing
import os
import tensorflow as tf


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

preprocessing.create_dataset(1, training=True)
preprocessing.create_dataset(1, training=False)

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
              loss=focal_loss(gamma=2.0),
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
          epochs=20,
          validation_split=0.1,
          verbose=2,
          callbacks=[tensorboard_callback])

model.save_weights(filepath=f"models/{MODEL_NAME}")
