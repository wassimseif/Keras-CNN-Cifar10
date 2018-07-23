import numpy as np
from keras.datasets import cifar10



class DataLoader():
    def load_data(self):

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        self.x_train = x_train
        self.y_train = y_train

        self.x_test =  x_test
        self.y_test = y_test

        print("Train samples:", x_train.shape, y_train.shape)
        print("Test samples:", x_test.shape, y_test.shape)


        self.classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

        return x_train, y_train, x_test, y_test , self.classes


    def visualize_some_data(self):

        cols = 8
        rows = 2
        fig = plt.figure(figsize=(2 * cols - 1, 2.5 * rows - 1))
        for i in range(cols):
            for j in range(rows):
                random_index = np.random.randint(0, len(self.y_train))
                ax = fig.add_subplot(rows, cols, i * rows + j + 1)
                ax.grid('off')
                ax.axis('off')
                ax.imshow(self.x_train[random_index, :])
                ax.set_title(self.classes[self.y_train[random_index, 0]])
        plt.show()

