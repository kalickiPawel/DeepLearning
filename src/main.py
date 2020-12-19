from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os

from laboratories import MLP
from src.utils import get_project_root

root = get_project_root()


def seconds_to_str(t):
    return "%02d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(round(t * 1000),), 1000, 60, 60]
    )


if __name__ == "__main__":
    output_path = os.path.join(root, 'out')
    json_path = os.path.join(output_path, 'results.json')

    activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'linear', 'relu', 'softmax']
    epochs = [10, 100, 1000]
    optimizers = ['adam', 'sgd', 'adadelta', 'adagrad', 'rmsprop']
    learning_rate = [0.01, 0.02, 0.05, 0.09, 0.1]

    test_cases = len(activation) * len(epochs) * len(optimizers) * len(learning_rate)
    print(f"Test cases: {test_cases}")

    # i = 0
    # summary_time = 0
    # experiments = []
    # for a in activation:
    #     for e in epochs:
    #         for o in optimizers:
    #             for lr in learning_rate:
    #                 t1 = time.time()
    #                 lab1 = MLP(activation=a, epochs=e, optimizer=o, learning_rate=lr, neurons=10, v=0)
    #                 result_time = time.time() - t1
    #                 print(f"Test case: {i + 1}/{test_cases} -> summary_time: {round(summary_time / 60, 2)} minutes")
    #                 r = lab1.check_model()
    #                 print(f"Done in time: {result_time} [seconds]")
    #
    #                 result = {
    #                     'id_probe': i,
    #                     'activation': a,
    #                     'epoch': e,
    #                     'neurons': 10,
    #                     'optimiser': o,
    #                     'learning_rate': lr,
    #                     'result': r,
    #                     'time': result_time
    #                 }
    #                 experiments.append(result)
    #                 with open(json_path, 'w') as fp:
    #                     json.dump({'experiments': experiments}, fp, indent=4)
    #                     fp.write('\n')
    #
    #                 i += 1
    #                 summary_time += result_time

    json_pd = pd.read_json(json_path)
    df = pd.DataFrame.from_records(json_pd['experiments'].values)
    summary_time = df.sum(axis=0, skipna=True)['time']

    df_results = pd.DataFrame(df["result"].to_list(), columns=['loss', 'accuracy'])
    df_results = df_results.assign(id_probe=df['id_probe'])

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(f"Results of {test_cases} test cases in {seconds_to_str(summary_time)}")
    ax1.plot(df['id_probe'], df['time'], 'b-')
    ax1.set_title('Time')

    ax2.plot(df_results['id_probe'], df_results['loss'], 'r-')
    ax2.set_title('Loss')

    ax3.plot(df_results['id_probe'], df_results['accuracy'], 'g-')
    ax3.set_title('Accuracy')

    plt.show()
