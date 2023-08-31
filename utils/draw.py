# coding: utf-8
import matplotlib.pyplot as plt
import os


def read_data(log_path):
    # Read experiment data for each experiment
    all_acc_list = []
    for exp in os.listdir(log_path):
        acc_list = read_exp_data(os.path.join(log_path, exp))
        all_acc_list.append((exp, acc_list))

    plt.figure(figsize=(10, 6))
    for exp, acc_list in all_acc_list:
        plt.plot(acc_list, label=exp)  # Plot the precision curve for each experiment

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Experiment Accuracy Comparison")
    plt.legend()
    plt.show()


def read_exp_data(exp_path):
    with open(os.path.join(exp_path, "accuracy.dat")) as f:
        acc_list = [float(i) for i in f.read().split()]
    return acc_list


if __name__ == '__main__':
    read_data("../log")
