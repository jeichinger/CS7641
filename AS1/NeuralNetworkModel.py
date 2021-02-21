import joblib
import os
from sklearn.neural_network import MLPClassifier
from BaseLearner import BaseLearner
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

OUTPUT = os.getcwd() + "/Output"

class NeuralNetworkModel(BaseLearner):

    def __init__(self, X_train, X_test, y_train, y_test, pipe, pre_processed_feature_names, class_names, dataset_name):

        self.model = MLPClassifier(random_state=1, verbose=51)

        super().__init__(X_train, X_test, y_train, y_test, pipe, self.model, pre_processed_feature_names, class_names, dataset_name)

        self.model.fit(self.X_train, self.y_train)
        self.model_params = {'random_state': 1,
                             'verbose': 51}

    def fit(self):
        super().model.fit(self.X_train, self.X_test)

    def predict(self, y):
        super().model.predict(y)

    def plot_epochs_vs_iterations(self):

        self.update_and_refit_model()

        loss = self.model.loss_curve_

        title = "Log Loss vs Iteration" + "\n" + self.dataset_name
        title_dic = {'fontsize': 6, 'fontweight': 'bold'}
        fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_xlabel("iterations", title_dic)
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Log Loss", title_dic)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax1.plot(loss, "r", linewidth=2)
        #    ax1.plot(param_range, valid_scores_mean, "b", linewidth=2, label="cross val")

        ax1.grid()
        plt.tight_layout()
        path = os.path.join(OUTPUT, self.dataset_name, "Artificial Neural Network")
        filename = "Log Loss vs Iteration" + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

    def generate_epoch_vs_configuration_learning_curve(self, param_range, configurations, iteration):

        training_scores_all = []
        valid_scores_all = []

        title = "Epochs vs CV F1 Score for (40,5) Hidden Layer Architecture." + " Iteration " + str(iteration) + "\n" + self.dataset_name
        title_dic = {'fontsize': 6, 'fontweight': 'bold'}
        fig, (ax1), = plt.subplots(1, 1, figsize=(5, 2))
        ax1.set_xlabel("Epochs", title_dic)
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Mean CV F1 Score", title_dic)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        for configuration in configurations:

            self.model_params["hidden_layer_sizes"] = configuration
            self.update_and_refit_model()

            train_scores, valid_scores = validation_curve(self.model, self.X_train, self.y_train, 'max_iter', param_range , cv=5,
                                                         scoring='f1', n_jobs=-1, verbose=51)

            train_scores_mean = np.mean(train_scores, axis=1)
            valid_scores_mean = np.mean(valid_scores, axis=1)

            training_scores_all.append(train_scores_mean)
            valid_scores_all.append(valid_scores_mean)

            print(str(configuration) + ":" + "Training Error:" + str(train_scores_mean) + "Validation Error:" + str(valid_scores_mean))
            ax1.plot(param_range, valid_scores_mean, linewidth=1, label=str(configuration))

        ax1.legend(loc= 'lower right', fontsize=6, ncol=3)
        ax1.grid()
        plt.tight_layout()
        path = os.path.join(OUTPUT, self.dataset_name, "Artificial Neural Network")
        filename = title + "_" + "Artificial Neural Network" + "_" + self.dataset_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

    def update_and_refit_model(self):

        self.model = MLPClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)