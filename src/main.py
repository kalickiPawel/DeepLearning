from laboratories import MLP
import matplotlib.pyplot as plt
import pandas as pd
import time


if __name__ == "__main__":
    # activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'linear', 'relu', 'softmax']
    # epochs = [10, 100, 1000]
    # optimizers = ['adam', 'sgd', 'adadelta', 'adagrad', 'rmsprop']
    # learning_rate = [0.01, 0.05, 0.1, 0.2, 0.5]
    activation = ['sigmoid', 'hard_sigmoid']
    epochs = [10, 100]
    optimizers = ['adam', 'sgd']
    learning_rate = [0.01, 0.05]

    results = {'id_probe': [], 'activation': [], 'epoch': [], 'optimiser': [], 'learning_rate': [], 'result': [], 'time': []}

    test_cases = len(activation)*len(epochs)*len(optimizers)*len(learning_rate)
    print(f"Test cases: {test_cases}")

    i = 0
    summary_time = 0
    for a in activation:
        for e in epochs:
            for o in optimizers:
                for lr in learning_rate:
                    t1 = time.time()
                    lab1 = MLP(activation=a, epochs=e, optimizer=o, learning_rate=lr, v=0)
                    result_time = time.time() - t1
                    results['id_probe'].append(i)
                    results['activation'].append(a)
                    results['epoch'].append(e)
                    results['optimiser'].append(o)
                    results['learning_rate'].append(lr)
                    print(f"Test case: {i}/{test_cases} -> summary_time: {round(summary_time/60, 2)} minutes")
                    results['result'].append(lab1.check_model())
                    results['time'].append(result_time)
                    print(f"Done in time: {result_time} [seconds]")
                    i += 1
                    summary_time += result_time

    df = pd.DataFrame.from_dict(results)
    df_results = pd.DataFrame(df["result"].to_list(), columns=['loss', 'accuracy'])
    df_results = df_results.assign(id_probe=df['id_probe'])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(f"Results of {test_cases} in {round(summary_time/60, 2)} minutes")
    ax1.plot(df['id_probe'], df['time'], 'b-')
    ax1.set_title('Time')

    ax2.plot(df_results['id_probe'], df_results['loss'], 'r-')
    ax2.set_title('Loss')

    ax3.plot(df_results['id_probe'], df_results['accuracy'], 'g-')
    ax3.set_title('Accuracy')

    plt.show()
