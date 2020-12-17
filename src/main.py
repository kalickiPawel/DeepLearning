from laboratories import MLP

if __name__ == "__main__":
    activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'linear', 'relu', 'softmax']
    epochs = [10, 100, 1000]
    optimizers = ['adam', 'sgd', 'adadelta', 'adagrad', 'rmsprop']
    learning_rate = 0.01
    lab1 = MLP()
    res = lab1.check_model()
    print(f"Result: {res}")
