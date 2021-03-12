from mlrose_hiive.runners import nngs_runner
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import mlrose_hiive.algorithms.gd
from mlrose_hiive.neural import NeuralNetwork
from DataPreprocessor import preprocess_data
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import NeuralNetworkLearner

DATA_PATH = os.getcwd() + "/Data"
OUTPUT_PATH = os.getcwd() + "/Output"
NEURAL_NETWORK_PATH = OUTPUT_PATH + "/NN/"
GRAD_DESCENT_PATH = "/GD"


def run_neural_network_optimization_experiment(dataset_name):
    np.random.seed(0)

    # Load the data.
    dataset_csv_path = os.path.join(DATA_PATH, dataset_name)
    dataset = pd.read_csv(dataset_csv_path)

    X = dataset.drop("class", axis=1)
    y = dataset["class"].copy().tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pipe = preprocess_data(X_train)
    X_train_transformed = pipe.fit_transform(X_train)

    model_params = {'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],
                    'max_iters': [1000],
                    'activation': [mlrose_hiive.relu]
                    }

    nn = NeuralNetwork(hidden_nodes=[12, 2],
                       activation='relu',
                       algorithm='gradient_descent',
                       max_iters=1500,
                       bias=True,
                       is_classifier=True,
                       early_stopping=False,
                       random_state=np.random.randint(100),
                       curve=True)

    nn_gd_learner = NeuralNetworkLearner.NeuralNetworkLearner(X_train_transformed, X_test, y_train, y_test, pipe, nn,
                                                              'GD')

    nn_gd_learner.update_and_refit_model()
    #nn_gd_learner.tune_hyper_parameter('learning_rate', [0.0001, 0.0003, 0.0004, 0.0005])

    nn_gd_learner.model.learning_rate = 0.0004

    # generate the learning curves.
    #nn_gd_learner.update_and_refit_model()
    #nn_gd_learner.generate_learning_curves("Learning Curve")


    temperature_list = [1, 10, 50, 100, 500, 1000]
    decay_list = [mlrose_hiive.GeomDecay, mlrose_hiive.ExpDecay, mlrose_hiive.ArithDecay]
    temperatures = [d(init_temp=t) for t in temperature_list for d in decay_list]

    # Simulated annealing
    sa_param_grid = {'schedule': temperatures,
                     'learning_rate': [0.001, 0.003, 0.005, 0.007, 0.009],
                     'max_iters': [100, 250, 500, 750, 1200, 1500]
                     }


    sa_nn = NeuralNetwork(hidden_nodes=[12, 2],
                          activation='relu',
                          algorithm='simulated_annealing',
                          max_iters=1500,
                          bias=True,
                          is_classifier=True,
                          early_stopping=True,
                          max_attempts=250,
                          random_state=np.random.randint(100),
                          curve=True)

    nn_sa_learner = NeuralNetworkLearner.NeuralNetworkLearner(X_train_transformed, X_test, y_train, y_test, pipe, sa_nn,
                                                              'SA')
    nn_sa_learner.update_and_refit_model()
    nn_sa_learner.find_hyper_params_coarse(sa_param_grid)

    #Randomized Hill Climbing
    rhc_param_grid = {'restarts': [10, 25, 50, 75],
                      'max_iters': [100, 250, 500, 750, 1200, 1500],
                      }

    rhc_nn = NeuralNetwork(hidden_nodes=[12, 2],
                          activation='relu',
                          algorithm='random_hill_climb',
                          max_iters=1500,
                          bias=True,
                          is_classifier=True,
                          early_stopping=True,
                          max_attempts=250,
                          random_state=np.random.randint(100),
                          curve=True)

    nn_rhc_learner = NeuralNetworkLearner.NeuralNetworkLearner(X_train_transformed, X_test, y_train, y_test, pipe, rhc_nn,
                                                              'RHC')
    nn_rhc_learner.update_and_refit_model()
    nn_rhc_learner.find_hyper_params_coarse(rhc_param_grid)


    ga_param_grid = {'mutation_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'max_iters': [100, 250, 500, 750, 1200, 1500],
                      }

    ga_nn = NeuralNetwork(hidden_nodes=[12, 2],
                          activation='relu',
                          algorithm='genetic_alg',
                          max_iters=1500,
                          bias=True,
                          is_classifier=True,
                          early_stopping=True,
                          max_attempts=250,
                          random_state=np.random.randint(100),
                          curve=True)

    ga_nn_learner = NeuralNetworkLearner.NeuralNetworkLearner(X_train_transformed, X_test, y_train, y_test, pipe, ga_nn,
                                                              'GA')
    ga_nn_learner.update_and_refit_model()
    ga_nn_learner.find_hyper_params_coarse(ga_param_grid)


'''
Code adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
'''


def tune_hyper_parameter(model, param_name, param_range, model_type, X_train, y_train):
    train_scores, valid_scores = validation_curve(model, X_train, y_train, param_name, param_range, cv=5,
                                                  scoring='f1', n_jobs=-1, verbose=51)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    title = "MC Curve for " + param_name + " Iteration " + "\n" + model_type
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
    path = NEURAL_NETWORK_PATH + model_type + "/"
    filename = "MC_Curve_" + param_name + ".png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)
