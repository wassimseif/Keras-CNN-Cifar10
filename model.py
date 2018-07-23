
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense, Activation,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import  adamax
NUM_CLASSES = 10

INIT_LR = 5e-3

def model():

    model = Sequential ()
    model.add (Conv2D (16, (3, 3), input_shape=(32, 32, 3), padding="same"))

    model.add (LeakyReLU (0.1))

    model.add (Conv2D (32, (3, 3), padding="same"))

    model.add (LeakyReLU (0.1))

    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add (Dropout (0.25))

    model.add (Conv2D(32, (3, 3), padding="same"))
    model.add (LeakyReLU (0.1))
    model.add (Conv2D (64, (3, 3), padding="same"))
    model.add (LeakyReLU (0.1))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add (Flatten ())

    model.add (Dense (256))
    model.add (LeakyReLU (0.1))
    model.add (Dropout (0.5))
    model.add (Dense (NUM_CLASSES))
    model.add (Activation ("softmax"))

    model.compile (
        loss='categorical_crossentropy',  # we train 10-way classification
        optimizer=adamax(lr=INIT_LR),  # for SGD
        metrics=['accuracy']  # report accuracy during training
    )


    return model

