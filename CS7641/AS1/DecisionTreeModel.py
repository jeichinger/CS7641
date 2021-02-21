import joblib
from sklearn.tree import DecisionTreeClassifier
from BaseLearner import BaseLearner
from sklearn.tree import plot_tree
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT = os.getcwd() + "/Output"

class DecisionTreeModel(BaseLearner):

    def __init__(self, X_train, X_test, y_train, y_test, pipe, pre_processed_feature_names, class_names, dataset_name):

        self.model = DecisionTreeClassifier(random_state=1)

        super().__init__(X_train, X_test, y_train, y_test, pipe, self.model, pre_processed_feature_names, class_names, dataset_name)

        self.model.fit(self.X_train, self.y_train)
        self.model_params = {'random_state': 1}

    def fit(self):

        super().model.fit(self.X_train, self.X_test)

    def predict(self, y):
        super().model.predict(y)

    '''
    Code adapted from: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
    '''
    def post_prune(self):

        path = self.model.cost_complexity_pruning_path(self.X_train, self.y_train)

        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        title_dic = {'fontsize': 6, 'fontweight': 'bold'}
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        ax.set_xlabel("effective alpha", title_dic)
        ax.set_ylabel("total impurity of leaves", title_dic)
        ax.set_title("Total Impurity vs effective alpha for training set", title_dic)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.grid()
        plt.tight_layout()
        path = os.path.join(OUTPUT, self.dataset_name, "DecisionTree")
        filename = "Impurities vs Alpha" + "_" + "DecisionTree" + "_" + self.dataset_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

        clfs = []
        for ccp_alpha in ccp_alphas:

            params = self.model_params
            params['ccp_alpha'] = ccp_alpha

            clf = DecisionTreeClassifier(**params)
            clf.fit(self.X_train, self.y_train)
            clfs.append(clf)

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax[0].set_xlabel("alpha", title_dic)
        ax[0].set_ylabel("number of nodes", title_dic)
        ax[0].set_title("Number of nodes vs alpha", title_dic)
        ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax[1].set_xlabel("alpha", title_dic)
        ax[1].set_ylabel("depth of tree",title_dic)
        ax[1].set_title("Depth vs alpha", title_dic)
        ax[0].grid()
        ax[1].grid()
        ax[0].tick_params(axis="x", labelsize=6)
        ax[0].tick_params(axis="y", labelsize=6)
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[1].tick_params(axis="x", labelsize=6)
        ax[1].tick_params(axis="y", labelsize=6)
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        fig.tight_layout()
        path = os.path.join(OUTPUT, self.dataset_name, "DecisionTree")
        filename = "Nodes and Depth vs Alpha" + "_" + "DecisionTree" + "_" + self.dataset_name + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

        train_scores = []
        valid_scores = []

        for clf in clfs:
            cv = cross_validate(clf, self.X_train, self.y_train, scoring='f1', return_train_score=True)
            train_scores.append(np.mean(cv['train_score']))
            valid_scores.append(np.mean(cv['test_score']))

        title = "MC Curve for MCC alpha" + "\n" + self.dataset_name
        title_dic = {'fontsize': 6, 'fontweight': 'bold'}
        fig, (ax1), = plt.subplots(1, 1, figsize=(3, 2))
        ax1.set_title(title, title_dic)
        ax1.set_ylabel("Mean F1 Score", title_dic)
        ax1.set_xlabel("alpha", title_dic)
        ax1.tick_params(axis="x", labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax1.plot(ccp_alphas, train_scores, 'r', linewidth=2, label="train")
        ax1.plot(ccp_alphas, valid_scores, 'b', linewidth=2, label="cross val")

        ax1.legend(loc='best', fontsize=6)
        ax1.grid()
        plt.tight_layout()
        path = os.path.join(OUTPUT, self.dataset_name, "DecisionTree")
        filename = "MC_Curve_" + "alpha" + ".png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

    def update_and_refit_model(self):

        self.model = DecisionTreeClassifier(**self.model_params)
        self.model.fit(self.X_train, self.y_train)

    def export_graph(self, filename):

        self.model.fit(self.X_train, self.y_train)
        plt.figure()

        plot_tree(self.model, feature_names=self.pre_processed_feature_names, class_names=self.class_names, rounded=True, filled=True, fontsize=4)
        plt.savefig(filename + ".eps", format='eps', bbox_inches='tight')



