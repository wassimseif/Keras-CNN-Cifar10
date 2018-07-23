from download_utils import DataLoader
import keras
from model import model
import keras.backend as K
import numpy as np
from TqdmProgressCallback import TqdmProgressCallback
#Hyper parameters
NUM_CLASSES = 10
INIT_LR = 5e-3
BATCH_SIZE = 32
EPOCHS = 14


data_loader = DataLoader()
x_train , y_train , x_test, y_test, classes  = data_loader.load_data()


x_train = (x_train / 255) - 0.5
x_test = (x_test / 255) - 0.5

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))

def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

model = model()


model.fit(
    x_train, y_train,  # prepared data
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler),
               LrHistory(),
               TqdmProgressCallback()],
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=0,
    initial_epoch= 0
)


print('SAVING THE MODEL')
model.save_weights("weights.h5")
print('MODEL SAVED')

# make test predictions

y_pred_test = model.predict_proba(x_test)

y_pred_test_classes = np.argmax(y_pred_test, axis=1)

y_pred_test_max_probas = np.max(y_pred_test, axis=1)