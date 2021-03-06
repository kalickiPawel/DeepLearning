from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop


class MLP:
    activation, optimizer = '', ''
    epochs, bs, neurons = 0, 0, 0
    lr = 0.0
    trainX, trainY = [], []
    testX, testY = [], []
    trainNorm, testNorm = [], []
    model = None

    def __init__(self, activation='softmax', epochs=10, optimizer='sgd', learning_rate=0.01, neurons=10, bs=256, v=1):
        self.activation = activation
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = learning_rate
        self.bs = bs
        self.neurons = neurons

        self.trainX, self.trainY, self.testX, self.testY = self.load_dataset()
        self.trainNorm, self.testNorm = self.normalize_data('x')
        self.model = self.get_model()
        self.fit_model(v)

    @staticmethod
    def load_dataset():
        # Load benchmark set MNIST
        (trainX, trainY), (testX, testY) = mnist.load_data()

        train_x = trainX.reshape((trainX.shape[0], 28, 28, 1))
        train_y = to_categorical(trainY)

        test_x = testX.reshape((testX.shape[0], 28, 28, 1))
        test_y = to_categorical(testY)
        return train_x, train_y, test_x, test_y

    def normalize_data(self, set_type):
        train, test = [], []
        if set_type == 'x':
            train = self.trainX.astype('float32')
            test = self.testX.astype('float32')
        elif set_type == 'y':
            train = self.trainY.astype('float32')
            test = self.testY.astype('float32')
        else:
            print("Probably this is wrong set")
        train, test = train / 255.0, test / 255.0
        return train, test

    def get_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(10, activation=self.activation, input_shape=(28 * 28, 1)))
        opt = None
        if self.optimizer == 'adam':
            opt = Adam(learning_rate=self.lr)
        elif self.optimizer == 'sgd':
            opt = SGD(learning_rate=self.lr)
        elif self.optimizer == 'adadelta':
            opt = Adadelta(learning_rate=self.lr)
        elif self.optimizer == 'adagrad':
            opt = Adagrad(learning_rate=self.lr)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=self.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit_model(self, v):
        self.model.fit(self.trainX, self.trainY, epochs=self.epochs, batch_size=self.bs, verbose=v)

    def check_model(self):
        return self.model.evaluate(self.testX, self.testY)
