class MLP:
    activation, optimizer = '', ''
    epochs, bs = 0, 0
    lr = 0.0

    def __init__(self, activation='softmax', epochs=10, optimizer='sgd', learning_rate=0.01, bs=256):
        self.activation = activation
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = learning_rate
        self.bs = bs
