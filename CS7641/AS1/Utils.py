import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import itertools
from matplotlib.ticker import FormatStrFormatter
import os

OUTPUT = os.getcwd() + "/Output"

def plot_svm_learners_on_same_curve(model_1, model_1_kernel, model_2, model_2_kernel, title):
    train_sizes, train_scores_1, validation_scores_1, fit_times_1, test_times_1 = learning_curve(model_1.model, model_1.X_train, model_1.y_train,
                                                                                train_sizes=np.linspace(0.1, 1, 10),
                                                                                return_times=True, cv=5, scoring='f1')

    train_sizes, train_scores_2, validation_scores_2, fit_times_2, test_times_2 = learning_curve(model_2.model, model_2.X_train, model_2.y_train,
                                                                                train_sizes=np.linspace(0.1, 1, 10),
                                                                                return_times=True, cv=5, scoring='f1')

    title_dic = {'fontsize': 6, 'fontweight': 'bold'}

    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 2))
    ax1.set_xlabel("% Of Training Data", title_dic)
    ax1.set_title(title + "\n" + model_1.dataset_name, title_dic)
    ax1.set_ylabel("Mean F1 Score", title_dic)
    ax1.tick_params(axis="x", labelsize=6)
    ax1.tick_params(axis="y", labelsize=6)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax1.plot(train_sizes / len(model_1.X_train), np.mean(train_scores_1, axis=1), "lightcoral", linewidth=2, label="train " + model_1_kernel + " kernel")
    ax1.plot(train_sizes / len(model_1.X_train), np.mean(validation_scores_1, axis=1), "firebrick", linewidth=2, label="cross val " + model_1_kernel + " kernel")

    ax1.plot(train_sizes / len(model_1.X_train), np.mean(train_scores_2, axis=1), "lightblue", linewidth=2, label="train " + model_2_kernel + " kernel")
    ax1.plot(train_sizes / len(model_1.X_train), np.mean(validation_scores_2, axis=1), "steelblue", linewidth=2,
             label="cross val " + model_2_kernel + " kernel")

    # Put a legend below current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)

    # ax2.set_xlabel("% Of Training Data", title_dic)
    # ax2.set_title("Train Time as a Function of Training Examples\n" + model_type + " - " + self.dataset_name, title_dic)
    # ax2.set_ylabel("Time(seconds)",title_dic)
    # ax2.tick_params(axis="x", labelsize=6)
    # ax2.tick_params(axis="y", labelsize=6)
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # ax2.plot(train_sizes / len(self.X_train), np.mean(fit_times, axis=1), "b", linewidth=2)

    ax1.grid()
    # ax2.grid()
    plt.tight_layout()

    path = os.path.join(OUTPUT, model_1.dataset_name)
    filename = title + "_" + "_" + model_1.dataset_name + ".png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)

    return np.mean(fit_times_1, axis=1), np.mean(test_times_1), np.mean(fit_times_2, axis=1), np.mean(test_times_2)

def plot_training_times(svm_poly_train_time, svm_poly_test_time, svm_rbf_train_time, svm_rbf_test_time, dt_train_time, dt_test_time, nn_train_time, nn_test_time, booster_train_time, booster_test_time, knn_train_time, knn_test_time, train_sizes, train_data_length, dataset_name):
    title_dic = {'fontsize': 6, 'fontweight': 'bold'}

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 2))
    ax1.set_xlabel("% Of Training Data", title_dic)
    ax1.set_title("Training Time" + "\n" + dataset_name, title_dic)
    ax1.set_ylabel("Average Time (ms) Log Scale", title_dic)
    ax1.set_yscale('log')
    ax1.tick_params(axis="x", labelsize=6)
    ax1.tick_params(axis="y", labelsize=6)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax1.plot(train_sizes / train_data_length, svm_poly_train_time*1000, linewidth=2, label="svm poly kernel")
    ax1.plot(train_sizes / train_data_length, svm_rbf_train_time*1000, linewidth=2, label="svm rbf kernel")
    ax1.plot(train_sizes / train_data_length, dt_train_time*1000, linewidth=2, label="decision tree")
    ax1.plot(train_sizes / train_data_length, nn_train_time*1000, linewidth=2, label="neural network")
    ax1.plot(train_sizes / train_data_length, booster_train_time*1000, linewidth=2, label="adaboost")
    ax1.plot(train_sizes / train_data_length, knn_train_time*1000, linewidth=2, label="k-nn")

    ax2.set_xlabel("Time (ms) Log Scale", title_dic)
    ax2.set_title("Test Time" + "\n" + dataset_name, title_dic)
    ax2.set_ylabel("Learners", title_dic)
    ax2.tick_params(axis="x", labelsize=6)
    ax2.tick_params(axis="y", labelsize=6)
    ax2.set_xscale('log')

    ax1.legend(loc='best', fontsize=6)

    x = ['svm poly kernel', 'svm rbf kernel', 'decision tree', 'neural network', 'adaboost', 'k-nn']

    x_pos = [i for i, _ in enumerate(x)]

    ax2.barh(x_pos, [svm_poly_test_time*1000, svm_rbf_test_time*1000, dt_test_time*1000, nn_test_time*1000, booster_test_time*1000, knn_test_time*1000])
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(x)

    ax1.grid()
    ax2.grid()
    plt.tight_layout()

    path = os.path.join(OUTPUT, dataset_name)
    filename = "Train and test times" + "_" + dataset_name + ".png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)
