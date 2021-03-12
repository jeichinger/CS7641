import pandas as pd
import numpy as np
import os
import sys
import time
import ContinuousPeaksExperiment
import KnapSackExperiment
import FFExperiment
import RunNeuralNetworkExperiment

import mlrose_hiive
LOG_PATH = os.getcwd() + "/Logs"

def main(find_params=True, run_cp_problem=False, run_knapsack_problem=False, run_ff_problem=False, run_nn=True, run_rhc=True,
         run_sa=True, run_ga=True, run_mimic=True):

    #sys.stdout = open(os.path.join(LOG_PATH, 'log' + time.strftime("%Y%m%d-%H%M%S") + ".txt"), 'w+')

    if run_cp_problem:
        ContinuousPeaksExperiment.run_continuous_peaks(True, True, True, True, True)

    if run_knapsack_problem:
        KnapSackExperiment.run_knapsack_experiment(True, True, True, True, True)

    if run_ff_problem:
        FFExperiment.run_flip_flop_experiment(True, True, True, True, True)

    if run_nn:
        RunNeuralNetworkExperiment.run_neural_network_optimization_experiment("diabetic.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Args for experiment')
    parser.add_argument('--dataset1', metavar='path', required=False, default="earthquake_processed.csv",
                        help='The path to dataset1')
    parser.add_argument('--dataset2', metavar='path', required=False, default="diabetic.csv",
                        help='The path to dataset2')
    args = parser.parse_args()

    main()
