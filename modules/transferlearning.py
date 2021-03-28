from keras import optimizers
from keras.applications import vgg16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.callbacks import TensorBoard

from focal_loss import focal_loss


def gennet_transfer_learning(x_shape, use_focal=False):

    # Load a pretrained model
    pre_trained = vgg16.VGG16(include_top=False, input_shape=x_shape)

    # Set pretrained layers to non-trainable
    for layer in pre_trained.layers:
        layer.trainable = False

    # Add new classifier layers, specific to the task
    flatten_layer = Flatten()(pre_trained.layers[-1].output)
    classify_layer = Dense(1024, activation='relu')(flatten_layer)
    output_layer = Dense(3, activation='softmax')(classify_layer)

    # Define new model and build on pretrained
    model = Model(inputs=pre_trained.inputs, outputs=output_layer)

    model.compile(optimizer=optimizers.Adam(),
                  loss=focal_loss if use_focal else "categorical_crossentropy",
                  metrics=['binary_accuracy'])

    # Summarize
    #model.summary()
    return model
