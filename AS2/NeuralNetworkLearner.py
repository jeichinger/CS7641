from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
from mlrose_hiive.neural import NeuralNetwork
import itertools

OUTPUT_PATH = os.getcwd() + "/Output/NN/"

class NeuralNetworkLearner:

    def __init__(self, X_train, X_test, y_train, y_test, pipe, model, model_type):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pipe = pipe
        self.model = model
        self.model_type = model_type
        self.model_params = {'random_state': 1}

    def update_and_refit_model(self):

        #self.model = NeuralNetwork(**self.model_params)
        self.model.fit(self.X_train, self.y_train)


    def run_cross_val_on_test_set(self):

        x_transformed = self.pipe.transform(self.X_test)

        y_pred = self.model.predict(x_transformed)

        score = f1_score(self.y_test, y_pred)

        print("-------------------------------------------------------------------------------------------------------", flush=True)
        print("F1 Score On Test Set: " + str(f1_score(self.y_test, y_pred)), flush=True)
        print("-------------------------------------------------------------------------------------------------------", flush=True)

        return score
    '''
    Adapted from Oreilly Hands-On Machine Learning with Python, Keras, and Tensorflow by Geron
    '''
    def generate_learning_curves(self, title):

        train_sizes, train_scores, validation_scores, fit_times, score_times = learning_curve(self.model, self.X_train, self.y_train, train_sizes=np.linspace(0.1, 1, 10), return_times=True, cv=5, scoring='f1')

        title_dic = {'fontsize': 7, 'fontweight':'bold'}

        fig, (ax1) = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_xlabel("% Of Training Data", title_dic)
        ax1.set_title(title + "\n" + self.model_type, title_dic)
        ax1.set_ylabel("Mean F1 Score", title_dic)
        ax1.tick_params(axis="x", labelsize=7)
        ax1.tick_params(axis="y", labelsize=7)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax1.plot(train_sizes/len(self.X_train), np.mean(train_scores, axis=1), "r", linewidth=2, label="train")
        ax1.plot(train_sizes/len(self.X_train), np.mean(validation_scores, axis=1), "b", linewidth=2, label="cross val")
        #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #          fancybox=True, ncol=5, fontsize=6)
        ax1.legend(loc='best', fontsize=6)

        #ax2.set_xlabel("% Of Training Data", title_dic)
        #ax2.set_title("Train Time as a Function of Training Examples\n" + model_type + " - " + self.dataset_name, title_dic)
        #ax2.set_ylabel("Time(seconds)",title_dic)
        #ax2.tick_params(axis="x", labelsize=6)
        #ax2.tick_params(axis="y", labelsize=6)
        #ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        #ax2.plot(train_sizes / len(self.X_train), np.mean(fit_times, axis=1), "b", linewidth=2)

        ax1.grid()
        #ax2.grid()
        plt.tight_layout()

        path = OUTPUT_PATH + '/' + self.model_type + "/"
        filename = title + "_" + self.model_type + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)
        return np.mean(fit_times, axis=1), np.mean(score_times), train_sizes
       # plt.show()

    '''
    Code adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    '''
    def tune_hyper_parameter(self, param_name, param_range):

        train_scores, valid_scores = validation_curve(self.model, self.X_train, self.y_train, param_name, param_range, cv=5,
                                                      scoring='f1', n_jobs=-1, verbose=51)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        title = "MC Curve for " + param_name + "\n" + self.model_type
        title_dic = {'fontsize': 7, 'fontweight': 'bold'}
        fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Mean F1 Score", title_dic)
        ax1.tick_params(axis="x", labelsize=7)
        ax1.tick_params(axis="y", labelsize=7)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax1.set_xlabel(param_name, title_dic)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkred", lw=2)
        plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.2,
                         color="navy", lw=2)

        ax1.plot(param_range, train_scores_mean, "r", linewidth=2, label="train")
        ax1.plot(param_range, valid_scores_mean, "b", linewidth=2, label="cross val")

        if param_name == 'max_iter':
            ax1.set_xlabel("iterations", title_dic)

        ax1.legend(loc='best', fontsize=6)
        ax1.grid()
        plt.tight_layout()
        path = OUTPUT_PATH + '/' + self.model_type + "/"
        filename = "MC_Curve_" + param_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

    def find_hyper_params_coarse(self, param_grid):

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1', return_train_score=True, n_jobs=-1, verbose=51)

        grid_search.fit(self.X_train, self.y_train)

        cvres = grid_search.cv_results_

        for f1_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(f1_score, params)

        print("-------------------------------------------------------------------------------------------------------")
        print("Best params found during coarse grid search of " + self.model_type + " parameters for " + ":\n" + str(grid_search.best_params_))
        print("-------------------------------------------------------------------------------------------------------")
        self.model = grid_search.best_estimator_
        self.model_params.update(grid_search.best_params_)

    def find_hyper_params_fine(self, param_grid, model_type):

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1', return_train_score=True, n_jobs=-1, verbose=51)

        grid_search.fit(self.X_train, self.y_train)

        cvres = grid_search.cv_results_

        for f1_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(f1_score, params)

        print("-------------------------------------------------------------------------------------------------------")
        print("Best params found during fine grid search of " + model_type + " parameters for " + self.dataset_name + ":\n" + str(grid_search.best_params_))
        print("-------------------------------------------------------------------------------------------------------")
        self.model = grid_search.best_estimator_

        self.model_params.update(grid_search.best_params_)
