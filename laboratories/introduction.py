from keras.datasets import mnist
from keras.utils import to_categorical


class MLP:
    activation, optimizer = '', ''
    epochs, bs = 0, 0
    lr = 0.0
    trainX, trainY = [], []
    testX, testY = [], []

    def __init__(self, activation='softmax', epochs=10, optimizer='sgd', learning_rate=0.01, bs=256):
        self.activation = activation
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = learning_rate
        self.bs = bs

        self.trainX, self.trainY, self.testX, self.testY = self.load_dataset()

    @staticmethod
    def load_dataset():
        # Load benchmark set MNIST
        (trainX, trainY), (testX, testY) = mnist.load_data()

        train_x = trainX.reshape((trainX.shape[0], 28, 28, 1))
        train_y = to_categorical(trainY)

        test_x = testX.reshape((testX.shape[0], 28, 28, 1))
        test_y = to_categorical(testY)
        return train_x, train_y, test_x, test_y
