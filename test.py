from download_utils import DataLoader
import keras
from model import model
import numpy as np
from sklearn.metrics import  accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

#Hyper parameters
NUM_CLASSES = 10
INIT_LR = 5e-3
BATCH_SIZE = 32
EPOCHS = 14


data_loader = DataLoader()
x_train , y_train , x_test, y_test, classes  = data_loader.load_data()


x_train2 = (x_train / 255) - 0.5
x_test2 = (x_test / 255) - 0.5

y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)


model = model()



model.load_weights("weights/weights.h5")



# make test predictions

y_pred_test = model.predict_proba(x_test)

y_pred_test_classes = np.argmax(y_pred_test, axis=1)

y_pred_test_max_probas = np.max(y_pred_test, axis=1)

plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_pred_test_classes))
plt.xticks(np.arange(10), classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), classes, fontsize=12)
plt.colorbar()
plt.show()
print("Test accuracy:", accuracy_score(y_test, y_pred_test_classes))


cols = 8
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = classes[y_pred_test_classes[random_index]]
        true_label = classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\ntrue: {}".format(
               pred_label, true_label
        ))
plt.show()