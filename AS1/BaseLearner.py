from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import itertools

OUTPUT = os.getcwd() + "/Output"

class BaseLearner:

    def __init__(self, X_train, X_test, y_train, y_test, pipe, model, pre_processed_feature_names, class_names, dataset_name):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pipe = pipe
        self.pre_processed_feature_names = pre_processed_feature_names
        self.class_names = class_names
        self.dataset_name = dataset_name
        self.model = model
        self.model_params = {'random_state': 1}


    def run_cross_val_on_test_set(self, confusion_matrix_title):

        x_transformed = self.pipe.transform(self.X_test)

        y_pred = self.model.predict(x_transformed)

        score = f1_score(self.y_test, y_pred)

        print("-------------------------------------------------------------------------------------------------------", flush=True)
        print("F1 Score On Test Set: " + str(f1_score(self.y_test, y_pred)), flush=True)
        print("-------------------------------------------------------------------------------------------------------", flush=True)

        cmap = plt.cm.get_cmap("Blues")

        cm = confusion_matrix(self.y_test, y_pred)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        title_dic = {'fontsize': 9, 'fontweight':'bold'}

        fig1 = plt.figure(figsize=(2.5,2.5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(confusion_matrix_title + '\n' + self.dataset_name, title_dic)
        #plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45, fontsize=9)
        plt.yticks(tick_marks, self.class_names, fontsize=9)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', title_dic)
        plt.xlabel('Predicted label', title_dic)

        path = os.path.join(OUTPUT, self.dataset_name)
        filename = confusion_matrix_title + "_" + "_" + self.dataset_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

        plt.close(fig1)

        return score
    '''
    Adapted from Oreilly Hands-On Machine Learning with Python, Keras, and Tensorflow by Geron
    '''
    def generate_learning_curves(self, title, model_type):

        train_sizes, train_scores, validation_scores, fit_times, score_times = learning_curve(self.model, self.X_train, self.y_train, train_sizes=np.linspace(0.1, 1, 10), return_times=True, cv=5, scoring='f1')

        title_dic = {'fontsize': 6, 'fontweight':'bold'}

        fig, (ax1) = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_xlabel("% Of Training Data", title_dic)
        ax1.set_title(title + "\n" + model_type + " - " + self.dataset_name, title_dic)
        ax1.set_ylabel("Mean F1 Score", title_dic)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
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

        path = os.path.join(OUTPUT, self.dataset_name, model_type)
        filename = title + "_" + model_type + "_" + self.dataset_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)
        return np.mean(fit_times, axis=1), np.mean(score_times), train_sizes
       # plt.show()

    '''
    Code adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    '''
    def tune_hyper_parameter(self, param_name, param_range, model_type, iteration=1):

        train_scores, valid_scores = validation_curve(self.model, self.X_train, self.y_train, param_name, param_range, cv=5,
                                                      scoring='f1', n_jobs=-1, verbose=51)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
        valid_scores_std = np.std(valid_scores, axis=1)

        title = "MC Curve for " + param_name + " Iteration " + str(iteration) + "\n" + self.dataset_name
        title_dic = {'fontsize': 6, 'fontweight': 'bold'}
        fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Mean F1 Score", title_dic)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        if param_name == 'hidden_layer_sizes':

            ax1.plot([str(item) for item in param_range], train_scores_mean, "r", linewidth=2, label="train")
            ax1.plot([str(item) for item in param_range], valid_scores_mean, "b", linewidth=2, label="cross val")
            ax1.set_xticklabels(param_range, rotation=90)

        elif param_name == 'base_estimator':

            ax1.plot([str(item.max_depth) for item in param_range], train_scores_mean, "r", linewidth=2, label="train")
            ax1.plot([str(item.max_depth) for item in param_range], valid_scores_mean, "b", linewidth=2, label="cross val")
            ax1.set_xlabel('max depth of base estimator', title_dic)


        elif param_name == 'metric':

            ax1.plot([str(item) for item in param_range], train_scores_mean, "r", linewidth=2, label="train")
            ax1.plot([str(item) for item in param_range], valid_scores_mean, "b", linewidth=2, label="cross val")
            ax1.set_xticklabels(param_range, rotation=90)
            ax1.set_xlabel(param_name, title_dic)

        elif param_name == 'criterion':

            ax1.plot([str(item) for item in param_range], train_scores_mean, "r", linewidth=2, label="train")
            ax1.plot([str(item) for item in param_range], valid_scores_mean, "b", linewidth=2, label="cross val")
            ax1.set_xticklabels(param_range, rotation=90)
            ax1.set_xlabel(param_name, title_dic)

        else:

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
        path = os.path.join(OUTPUT, self.dataset_name, model_type)
        filename = "MC_Curve_" + param_name + "_iteration_" + str(iteration) + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

    def find_hyper_params_coarse(self, param_grid, model_type):

        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1', return_train_score=True, n_jobs=-1, verbose=51)

        grid_search.fit(self.X_train, self.y_train)

        cvres = grid_search.cv_results_

        for f1_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(f1_score, params)

        print("-------------------------------------------------------------------------------------------------------")
        print("Best params found during coarse grid search of " + model_type + " parameters for " + self.dataset_name + ":\n" + str(grid_search.best_params_))
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
