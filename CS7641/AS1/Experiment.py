import pandas as pd
import numpy as np
import os
import DataPreprocessor
import sys
import time
import matplotlib.pyplot as plt
from DecisionTreeModel import DecisionTreeModel
from NeuralNetworkModel import NeuralNetworkModel
from AdaBoostClassifier import AdaBoostModel
from SVMModel import SVMModel
from KNNModel import KNNModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import Utils


DATA_PATH = os.getcwd() + "/Data"
LOG_PATH = os.getcwd() + "/Logs"

def main(dataset1="earthquake_processed.csv", dataset2="diabetic.csv", run_dt=True, run_nn=True, run_boost=True, run_svm=True, run_knn=True):

    sys.stdout = open(os.path.join(LOG_PATH, 'log' + time.strftime("%Y%m%d-%H%M%S") + ".txt"), 'w+')

    dt_param_grid_coarse = {
        # 'min_impurity_decrease':[0.0, 0.03, 0.9],
        'max_depth': [1, 100, 500],
        'min_samples_split': [0.01, 0.05],
        'min_samples_leaf': [0.01, 0.05],
        'max_features': [None, 'auto']
    }

    ann_param_grid_coarse = {
        'hidden_layer_sizes': [(5), (50), (100), (5, 5), (50, 50), (100, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
    }

    svm_rbf_kernel_param_grid_coarse = {
        'kernel':['rbf'],
        'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
        'C' : [0.1, 1, 10, 100, 1000]
    }


    if dataset1 != "":

        print("Loading dataset " + dataset1, flush=True)

        # Load the data.
        dataset_csv_path = os.path.join(DATA_PATH, dataset1)
        dataset = pd.read_csv(dataset_csv_path)

        X = dataset.drop("class", axis=1)
        y = dataset["class"].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        pipe = DataPreprocessor.preprocess_data(X_train)
        X_train_transformed = pipe.fit_transform(X_train)

        pre_processed_feature_names = X_train_transformed

        earthquake_svm_poly_score = None
        earthquake_svm_poly_train_times = None
        earthquake_svm_poly_test_times = None

        earthquake_svm_rbf_score = None
        earthquake_svm_rbf_train_times = None
        earthquake_svm_rbf_test_times = None

        earthquake_decision_tree_score = None
        earthquake_decision_tree_train_times = None
        earthquake_decision_tree_test_times = None

        earthquake_nn_score = None
        earthquake_nn_train_times = None
        earthquake_nn_test_times = None

        earthquake_booster_score = None
        earthquake_booster_train_times = None
        earthquake_booster_test_times = None

        earthquake_knn_score = None
        earthquake_knn_train_times = None
        earthquake_knn_test_times = None

        earthquake_train_sizes = None

        if run_svm:

            svm_poly_kernel_param_grid_coarse = {
                'kernel': ['poly'],
                'degree': [3],
                'gamma': [ 0.1, 0.01, 0.001, 0.0001],
                'C': [0.1, 1, 10, 100]
            }

            svm_poly = SVMModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                                ["No Damage ", "Destroyed"], "Earthquake")

            svm_rbf_kernel_param_grid_coarse = {
                'kernel': ['rbf'],
                'gamma': [1, 0.1],
                'C': [0.1, 1, 10, 100, 1000]
            }

            svm_rbf = SVMModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                               ["No Damage ", "Destroyed"], "Earthquake")

            svm_rbf.find_hyper_params_coarse(svm_rbf_kernel_param_grid_coarse, "SVMRbf")
            # Best params: C:1, gamma:0.1

            svm_poly.find_hyper_params_coarse(svm_poly_kernel_param_grid_coarse, "SVMPoly")
            # Best params: C:0.1 degree:3, gamma:0.1

            Utils.plot_svm_learners_on_same_curve(svm_poly, "poly", svm_rbf, "rbf", "Learning Curves After Coarse Grid Search")

            svm_poly.model_params['kernel'] = 'poly'
            svm_poly.update_and_refit_model()

            param_range = [2, 3, 4]
            svm_poly.tune_hyper_parameter('degree', param_range, "SVMPoly", 1)

            svm_poly.model_params['degree'] = 2
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 0.2, 50)
            svm_poly.tune_hyper_parameter('C', param_range, "SVMPoly", 1)

            svm_poly.model_params['C'] = 0.1
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 5, 25)
            svm_poly.tune_hyper_parameter('coef0', param_range, "SVMPoly", 1)

            svm_poly.model_params['coef0'] = 3
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 0.010, 25)
            svm_poly.tune_hyper_parameter('gamma', param_range, "SVMPoly", 1)

            svm_poly.model_params['gamma'] = .008
            svm_poly.update_and_refit_model()

            ############## RBF Kernel ##########################################

            svm_rbf.model_params['kernel'] = 'rbf'
            svm_rbf.update_and_refit_model()

            param_range = np.linspace(0.0, 0.05, 50)
            svm_rbf.tune_hyper_parameter('gamma', param_range, "SVMRbf", 1)

            svm_rbf.model_params['gamma'] = 0.008
            svm_rbf.update_and_refit_model()

            param_range = np.linspace(0, 5, 50)
            svm_rbf.tune_hyper_parameter('C', param_range, "SVMRbf", 1)

            svm_rbf.model_params['C'] = 0.9
            svm_rbf.update_and_refit_model()

            earthquake_svm_rbf_score = svm_rbf.run_cross_val_on_test_set("SVM RBF Kernel")
            earthquake_svm_poly_score = svm_poly.run_cross_val_on_test_set("SVM Poly Kernel")

            earthquake_svm_poly_train_times, earthquake_svm_poly_test_times, earthquake_svm_rbf_train_times, earthquake_svm_rbf_test_times = \
                Utils.plot_svm_learners_on_same_curve(svm_poly, "poly", svm_rbf, "rbf", "Final Learning Curves")

        if run_dt:

            print("running dt experiment", flush=True)
            dt = DecisionTreeModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Damage", "Destroyed"], "Earthquake")

            dt.generate_learning_curves("CV Learning Curve Before Any Pruning", "DecisionTree")

            print("Depth of decision tree before pruning" + str(dt.model.get_depth()), flush=True)
            print("Number of nodes before pruning" + str(dt.model.tree_.node_count), flush=True)

            print("finding coarse hyper params for pre-pruning.", flush=True)

            dt.find_hyper_params_coarse(dt_param_grid_coarse, "DecisionTree")

            param_range = ['gini', 'entropy']
            dt.tune_hyper_parameter('criterion', param_range, "DecisionTree")

            dt.model_params['criterion'] = 'gini'
            dt.update_and_refit_model()

            print("Depth of decision tree after pre-pruning: " + str(dt.model.get_depth()), flush=True)
            print("Number of nodes after pre-pruning: " + str(dt.model.tree_.node_count), flush=True)

            print("generating learning curves", flush=True)
            dt.generate_learning_curves("CV Learning Curve after Coarse Grid Search for Pre-Pruning","DecisionTree")

            print("Finding optimal depth of tree", flush=True)
            param_range = np.arange(1, 25, 1)
            dt.tune_hyper_parameter('max_depth', param_range, "DecisionTree")

            dt.model_params['max_depth'] = 6
            dt.update_and_refit_model()

            print("generating learning curves")
            dt.generate_learning_curves("CV Learning Curve after Refined Max Depth Search","DecisionTree")

            print("Finding optimal min sample leaves of tree")
            dt.tune_hyper_parameter('min_samples_leaf', param_range, "DecisionTree")

            dt.model_params['min_samples_leaf'] = 5
            dt.update_and_refit_model()

            print("Depth of decision tree after Refined-pruning: " + str(dt.model.get_depth()))
            print("Number of nodes after Refined-pruning: " + str(dt.model.tree_.node_count))

            print("generating learning curves")
            dt.generate_learning_curves("CV Learning Curve after Refined Max Depth and Min Sample Leaves","DecisionTree")

            # Post Pruning:
            # Reset Pre-pruning variables.
            dt.model_params['min_samples_leaf'] = 1
            dt.model_params['max_depth'] = None
            dt.model_params['min_samples_split'] = 2
            dt.update_and_refit_model()

            # Post Prune
            dt.post_prune()

            param_grid = {
                'max_depth': [4, 5, 6, 7, 8, 9, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
                'ccp_alpha': [0.0002, 0.0005, 0.001, 0.0015, 0.0020]
            }

            dt.find_hyper_params_fine(param_grid, "DecisionTree")

            earthquake_decision_tree_score = dt.run_cross_val_on_test_set("Decision Tree")

            print("generating learning curves")
            earthquake_decision_tree_train_times, earthquake_decision_tree_test_times, earthquake_train_sizes = dt.generate_learning_curves("CV Learning Curve Combining Pre and Post Pruning","DecisionTree")

        if run_nn:

            ann = NeuralNetworkModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Damage", "Destroyed"], "Earthquake")

            ann.model_params['solver'] = 'sgd'
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve Before Any Tuning", "Artificial Neural Network")

            ann.model_params['max_iter'] = 1000
            ann.update_and_refit_model()

            param_range = [0.0006, 0.0008, 0.001, 0.0012, 0.0014]
            ann.tune_hyper_parameter('learning_rate_init', param_range, "Artificial Neural Network")

            ann.model_params['learning_rate_init'] = 0.0010
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve After Tuning Learning Rate", "Artificial Neural Network")

            param_range = [10, 25, 40, 50, 75, 100, 125, 150, 200]
            ann.tune_hyper_parameter('hidden_layer_sizes', param_range, "Artificial Neural Network", 1)

            param_range = [33, 36, 39, 40, 41, 42, 45]
            ann.tune_hyper_parameter('hidden_layer_sizes', param_range, "Artificial Neural Network", 2)

            ann.model_params['hidden_layer_sizes'] = (40)
            ann.update_and_refit_model()

            param_range = [0.0008, 0.001, 0.0012, 0.0014, .0016, .0018, .002]
            ann.tune_hyper_parameter('learning_rate_init', param_range, "Artificial Neural Network", 2)

            ann.model_params['learning_rate_init'] = 0.0018
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve After Tuning First Hidden Layer",
                                        "Artificial Neural Network")

            print("Finding optimal momentum rate")
            param_range = [0.88, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95]
            ann.tune_hyper_parameter('momentum', param_range, "Artificial Neural Network", 1)

            ann.model_params['momentum'] = 0.89
            ann.update_and_refit_model()

            param_range = [40, (40,2), (40,3), (40,4), (40,5), (40,10), (40,15), (40,20), (40,25), (40,30), (40,40)]
            ann.tune_hyper_parameter('hidden_layer_sizes', param_range, "Artificial Neural Network", 3)

            param_range = np.arange(50, 1000, 100)
            ann.tune_hyper_parameter('max_iter', param_range, "Artificial Neural Network", 1)

            ann.plot_epochs_vs_iterations()

            ann.model_params['max_iter'] = 250
            ann.update_and_refit_model()

            earthquake_nn_score = ann.run_cross_val_on_test_set("Neural Network")

            earthquake_nn_train_times, earthquake_nn_test_times, _ =  ann.generate_learning_curves("Final CV Learning Curve",
                                        "Artificial Neural Network")

        if run_boost:

            booster = AdaBoostModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                                ["No Damage ", "Destroyed"], "Earthquake")

            booster.generate_learning_curves("Learning Curve Before Tuning", "AdaBoost")

            param_range = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4)]
            booster.tune_hyper_parameter('base_estimator', param_range, "AdaBoost")

            booster.model_params['base_estimator'] = DecisionTreeClassifier(max_depth=1)
            booster.update_and_refit_model()

            param_range = np.arange(1, 100, 1)
            booster.tune_hyper_parameter('n_estimators', param_range, "AdaBoost", 1)

            booster.model_params['n_estimators'] = 23
            booster.update_and_refit_model()

            booster.generate_learning_curves("Learning Curve After Tuning Number Estimators", "AdaBoost")

            param_range = np.linspace(0, 1, 25)
            booster.tune_hyper_parameter('learning_rate', param_range, "AdaBoost", 1)

            booster.model_params['learning_rate'] = 0.5
            booster.update_and_refit_model()

            earthquake_booster_score = booster.run_cross_val_on_test_set("AdaBoost")

            earthquake_booster_train_times, earthquake_booster_test_times, _ = booster.generate_learning_curves("Learning Curve After Tuning Learning Rate", "AdaBoost")

        if run_knn:
            knn = KNNModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Damage", "Destroyed"], "Earthquake")
            knn.generate_learning_curves("Learning Curve Before Tuning", "KNN")

            param_range = np.arange(1, 25, 1)
            knn.tune_hyper_parameter('n_neighbors', param_range, "KNN", 1)
            knn.model_params['n_neighbors'] = 24
            knn.update_and_refit_model()

            knn.generate_learning_curves("Learning Curve After Tuning K", "KNN")

            param_range = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
            knn.tune_hyper_parameter('metric', param_range, "KNN", 1)
            knn.model_params['metric'] = 'manhattan'
            knn.update_and_refit_model()

            earthquake_knn_score = knn.run_cross_val_on_test_set("K-NN")

            earthquake_knn_train_times, earthquake_knn_test_times, _ = knn.generate_learning_curves("Learning Curve After Tuning Distance Metric", "KNN")

        Utils.plot_training_times(earthquake_svm_poly_train_times,
                                  earthquake_svm_poly_test_times,
                                  earthquake_svm_rbf_train_times,
                                  earthquake_svm_rbf_test_times,
                                  earthquake_decision_tree_train_times,
                                  earthquake_decision_tree_test_times,
                                  earthquake_nn_train_times,
                                  earthquake_nn_test_times,
                                  earthquake_booster_train_times,
                                  earthquake_booster_test_times,
                                  earthquake_knn_train_times,
                                  earthquake_knn_test_times,
                                  earthquake_train_sizes,
                                  len(X_train),
                                  "Earthquake")

        print("Score of svm with poly kernel:" + str(earthquake_svm_poly_score))
        print("Score of svm with rbf kernel:" + str(earthquake_svm_rbf_score))
        print("Score of decision tree:" + str(earthquake_decision_tree_score))
        print("Score of neural networkl:" + str(earthquake_nn_score))
        print("Score of booster:" + str(earthquake_booster_score))
        print("Score of knn:" + str(earthquake_knn_score))


    if dataset2 != "":

        print("Loading dataset " + dataset2)

        # Load the data.
        dataset_csv_path = os.path.join(DATA_PATH, dataset2)
        dataset = pd.read_csv(dataset_csv_path)

        X = dataset.drop("class", axis=1)
        y = dataset["class"].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        pipe = DataPreprocessor.preprocess_data(X_train)
        X_train_transformed = pipe.fit_transform(X_train)

        pre_processed_feature_names = X_train_transformed

        diabetes_svm_poly_score = None
        diabetes_svm_poly_train_times = None
        diabetes_svm_poly_test_times = None

        diabetes_svm_rbf_score = None
        diabetes_svm_rbf_train_times = None
        diabetes_svm_rbf_test_times = None

        diabetes_decision_tree_score = None
        diabetes_decision_tree_train_times = None
        diabetes_decision_tree_test_times = None

        diabetes_nn_score = None
        diabetes_nn_train_times = None
        diabetes_nn_test_times = None

        diabetes_booster_score = None
        diabetes_booster_train_times = None
        diabetes_booster_test_times = None

        diabetes_knn_score = None
        diabetes_knn_train_times = None
        diabetes_knn_test_times = None

        diabetes_train_sizes = None

        if run_svm:

            svm_poly_kernel_param_grid_coarse = {
                'kernel': ['poly'],
                'degree': [3, 7],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'C': [0.1, 1, 10, 100, 1000]
            }

            svm_poly = SVMModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                                ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")

            svm_rbf_kernel_param_grid_coarse = {
                'kernel': ['rbf'],
                'gamma': [1, 0.1],
                'C': [0.1, 1, 10, 100, 1000]
            }

            svm_rbf = SVMModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                               ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")

            svm_rbf.find_hyper_params_coarse(svm_rbf_kernel_param_grid_coarse, "SVMRbf")

            svm_poly.find_hyper_params_coarse(svm_poly_kernel_param_grid_coarse, "SVMPoly")

            Utils.plot_svm_learners_on_same_curve(svm_poly, "poly", svm_rbf, "rbf", "Learning Curves After Coarse Grid Search")

            svm_poly.model_params['kernel'] = 'poly'
            svm_poly.update_and_refit_model()

            param_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11 , 12]
            svm_poly.tune_hyper_parameter('degree', param_range, "SVMPoly", 1)

            svm_poly.model_params['degree'] = 3
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 0.2, 50)
            svm_poly.tune_hyper_parameter('C', param_range, "SVMPoly", 1)

            svm_poly.model_params['C'] = 0.02
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 50, 50)
            svm_poly.tune_hyper_parameter('coef0', param_range, "SVMPoly", 1)

            svm_poly.model_params['coef0'] = 40
            svm_poly.update_and_refit_model()

            param_range = np.linspace(0.0, 0.25, 50)
            svm_poly.tune_hyper_parameter('gamma', param_range, "SVMPoly", 1)

            svm_poly.model_params['gamma'] = .050
            svm_poly.update_and_refit_model()

            ############## RBF Kernel ##########################################

            svm_rbf.model_params['kernel'] = 'rbf'
            svm_rbf.update_and_refit_model()

            param_range = np.linspace(0.0, 0.02, 50)
            svm_rbf.tune_hyper_parameter('gamma', param_range, "SVMRbf", 1)

            svm_rbf.model_params['gamma'] = 0.010
            svm_rbf.update_and_refit_model()

            param_range = np.linspace(-100, 300, 50)
            svm_rbf.tune_hyper_parameter('C', param_range, "SVMRbf", 1)

            svm_rbf.model_params['C'] = 125
            svm_rbf.update_and_refit_model()

            diabetes_svm_rbf_score = svm_rbf.run_cross_val_on_test_set("SVM RBF Kernel")
            diabetes_svm_poly_score = svm_poly.run_cross_val_on_test_set("SVM Poly Kernel")

            diabetes_svm_poly_train_times, diabetes_svm_poly_test_times, diabetes_svm_rbf_train_times, diabetes_svm_rbf_test_times = Utils.plot_svm_learners_on_same_curve(svm_poly, "poly", svm_rbf, "rbf", "Final Learning Curves")

        if run_dt:

            print("running dt experiment")
            dt = DecisionTreeModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")

            dt.generate_learning_curves("CV Learning Curve Before Any Pruning", "DecisionTree")

            print("Depth of decision tree before pruning" + str(dt.model.get_depth()))
            print("Number of nodes before pruning" + str(dt.model.tree_.node_count))

            print("finding coarse hyper params for pre-pruning.")

            dt.find_hyper_params_coarse(dt_param_grid_coarse, "DecisionTree")

            print("Depth of decision tree after pre-pruning: " + str(dt.model.get_depth()))
            print("Number of nodes after pre-pruning: " + str(dt.model.tree_.node_count))

            print("generating learning curves")
            dt.generate_learning_curves("CV Learning Curve after Coarse Grid Search for Pre-Pruning","DecisionTree")

            param_range = ['gini', 'entropy']
            dt.tune_hyper_parameter('criterion', param_range, "DecisionTree")

            # Uncomment to plot range of max_depth parameter.
            print("Finding optimal depth of tree")
            param_range = np.arange(1, 25, 1)
            dt.tune_hyper_parameter('max_depth', param_range, "DecisionTree")

            dt.model_params['max_depth'] = 20
            dt.update_and_refit_model()

            print("generating learning curves")
            dt.generate_learning_curves("CV Learning Curve after Refined Max Depth Search","DecisionTree")

            print("Finding optimal min sample leaves of tree")
            param_range = np.arange(1, 25, 1)
            dt.tune_hyper_parameter('min_samples_leaf', param_range, "DecisionTree")

            dt.model_params['min_samples_leaf'] = 15
            dt.update_and_refit_model()

            print("Depth of decision tree after Refined-pruning: " + str(dt.model.get_depth()))
            print("Number of nodes after Refined-pruning: " + str(dt.model.tree_.node_count))

            print("generating learning curves")
            dt.generate_learning_curves("CV Learning Curve after Refined Max Depth and Min Sample Leaves","DecisionTree")

            # Post Pruning:
            # Reset Pre-pruning variables.
            dt.model_params['min_samples_leaf'] = 1
            dt.model_params['max_depth'] = None
            dt.model_params['min_samples_split'] = 2
            dt.update_and_refit_model()

            # Post Prune
            dt.post_prune()

            param_grid = {
                'max_depth': [ 10, 15, 20],
                'min_samples_leaf': [14, 15, 16, 17, 18],
                'ccp_alpha': [0.000, 0.002, 0.003, 0.004, 0.005]
            }

            dt.find_hyper_params_fine(param_grid, "DecisionTree")

            diabetes_decision_tree_score = dt.run_cross_val_on_test_set("DecisionTree")

            print("generating learning curves")
            diabetes_decision_tree_train_times, diabetes_decision_tree_test_times, diabetes_train_sizes = dt.generate_learning_curves("CV Learning Curve Combining Pre and Post Pruning","DecisionTree")

        if run_nn:

            print("running nn experiment")
            ann = NeuralNetworkModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")

            ann.model_params['solver'] = 'sgd'
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve Before Any Tuning", "Artificial Neural Network")

            ann.model_params['max_iter'] = 1000
            ann.update_and_refit_model()

            print("Finding optimal learning rate")
            param_range = [.0009, 0.0010, 0.0013, 0.0015, 0.0017, 0.002, 0.003, 0.004, .005, .006, .007]
            ann.tune_hyper_parameter('learning_rate_init', param_range, "Artificial Neural Network")

            ann.model_params['learning_rate_init'] = 0.004
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve After Tuning Initial Learning Rate", "Artificial Neural Network")

            print("Finding optimal hidden layers")
            param_range = [(4),(6), (8), (10), (12), (15), (20), (25), (100)]
            ann.tune_hyper_parameter('hidden_layer_sizes', param_range, "Artificial Neural Network", 1)

            ann.model_params['hidden_layer_sizes'] = (12)
            ann.update_and_refit_model()

            print("Finding optimal learning rate")
            param_range = [ 0.0010, 0.0013, 0.0015, 0.0017, 0.002, 0.003, 0.004, .005, .006,.007,.008]
            ann.tune_hyper_parameter('learning_rate_init', param_range, "Artificial Neural Network", 2)

            ann.model_params['learning_rate_init'] = 0.006
            ann.update_and_refit_model()

            ann.generate_learning_curves("CV Learning Curve After Tuning 1st Hidden Layer", "Artificial Neural Network")

            print("Finding optimal momentum rate")
            param_range = [0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
            ann.tune_hyper_parameter('momentum', param_range, "Artificial Neural Network", 1)

            print("Finding optimal hidden layers")
            param_range = [12,(12,2),(12,4), (12,6), (12,8), (12,10), (12,12), (12,14), (12,16), (15,15), (20,20),(50,50)]
            ann.tune_hyper_parameter('hidden_layer_sizes', param_range, "Artificial Neural Network", 2)

            ann.model_params['hidden_layer_sizes'] = (12,2)
            ann.update_and_refit_model()

            param_range = np.arange(50, 1200, 100)
            ann.tune_hyper_parameter('max_iter', param_range, "Artificial Neural Network", 1)

            ann.model_params['tol'] = 0.000001
            ann.update_and_refit_model()

            ann.plot_epochs_vs_iterations()

            print("Finding optimal learning rate")
            param_range = [ 0.0010, 0.0013, 0.0015, 0.0017, 0.002, 0.003, 0.004, .005, .006,.007,.008]
            ann.tune_hyper_parameter('learning_rate_init', param_range, "Artificial Neural Network", 3)

            diabetes_nn_score = ann.run_cross_val_on_test_set("Neural Network")

            diabetes_nn_train_times, diabetes_nn_test_times, _ = ann.generate_learning_curves("Final CV Learning Curve", "Artificial Neural Network")
        if run_boost:
            booster = AdaBoostModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names,
                                ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")

            booster.generate_learning_curves("Learning Curve Before Tuning", "AdaBoost")

            param_range = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4)]
            booster.tune_hyper_parameter('base_estimator', param_range, "AdaBoost")

            booster.model_params['base_estimator'] = DecisionTreeClassifier(max_depth=2)
            booster.update_and_refit_model()

            param_range = np.arange(1, 100, 1)
            booster.tune_hyper_parameter('n_estimators', param_range, "AdaBoost", 1)

            booster.model_params['n_estimators'] = 19
            booster.update_and_refit_model()

            booster.generate_learning_curves("Learning Curve After Tuning Number Estimators", "AdaBoost")

            param_range = np.linspace(0, 1, 25)
            booster.tune_hyper_parameter('learning_rate', param_range, "AdaBoost", 1)

            booster.model_params['learning_rate'] = 0.3
            booster.update_and_refit_model()

            diabetes_booster_score = booster.run_cross_val_on_test_set("AdaBoost")

            diabetes_booster_train_times, diabetes_booster_test_times, _ = booster.generate_learning_curves("Learning Curve After Tuning Learning Rate", "AdaBoost")

        if run_knn:
            knn = KNNModel(X_train_transformed, X_test, y_train, y_test, pipe, pre_processed_feature_names, ["No Signs ", "Signs of DR"], "Diabetic Retinopathy")
            knn.generate_learning_curves("Learning Curve Before Tuning", "KNN")

            param_range = np.arange(1, 25, 1)
            knn.tune_hyper_parameter('n_neighbors', param_range, "KNN", 1)

            knn.model_params['n_neighbors'] = 17
            knn.update_and_refit_model()

            knn.generate_learning_curves("Learning Curve After Tuning K", "KNN")

            param_range = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
            knn.tune_hyper_parameter('metric', param_range, "KNN", 1)

            knn.model_params['metric'] = 'braycurtis'
            knn.update_and_refit_model()

            diabetes_knn_score = knn.run_cross_val_on_test_set("K-NN")

            diabetes_knn_train_times, diabetes_knn_test_times, _ = knn.generate_learning_curves("Learning Curve After Tuning Distance Metric", "KNN")

        Utils.plot_training_times(diabetes_svm_poly_train_times,
                                  diabetes_svm_poly_test_times,
                                  diabetes_svm_rbf_train_times,
                                  diabetes_svm_rbf_test_times,
                                  diabetes_decision_tree_train_times,
                                  diabetes_decision_tree_test_times,
                                  diabetes_nn_train_times,
                                  diabetes_nn_test_times,
                                  diabetes_booster_train_times,
                                  diabetes_booster_test_times,
                                  diabetes_knn_train_times,
                                  diabetes_knn_test_times,
                                  diabetes_train_sizes,
                                  len(X_train),
                                  "Diabetic Retinopathy")

        print("Score of svm with poly kernel:" + str(diabetes_svm_poly_score))
        print("Score of svm with rbf kernel:" + str(diabetes_svm_rbf_score))
        print("Score of decision tree:" + str(diabetes_decision_tree_score))
        print("Score of neural networkl:" + str(diabetes_nn_score))
        print("Score of booster:" + str(diabetes_booster_score))
        print("Score of knn:" + str(diabetes_knn_score))

    sys.stdout.flush()
    sys.stdout.close()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Args for experiment')
    parser.add_argument('--dataset1', metavar='path', required=False, default="earthquake_processed.csv",
                        help='The path to dataset1')
    parser.add_argument('--dataset2', metavar='path', required=False, default="diabetic.csv",
                        help='The path to dataset2')
    args = parser.parse_args()

    main(dataset1=args.dataset1,
         dataset2=args.dataset2)
