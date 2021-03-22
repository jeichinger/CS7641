import pandas as pd
import numpy as np
import os
import DataPreprocessor
import sys
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import InterclusterDistance

DATA_PATH = os.getcwd() + "/Data"
LOG_PATH = os.getcwd() + "/Logs"
OUTPUT = os.getcwd() + "/Output"

def main(dataset2="diabetic.csv", dataset1="earthquake_processed.csv"):

    #sys.stdout = open(os.path.join(LOG_PATH, 'log' + time.strftime("%Y%m%d-%H%M%S") + ".txt"), 'w+')

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

        k_values = np.arange(2, 10, 1)
        inertia_values = []
        silhouette_scores = []
        for k in k_values:

            kmeans = KMeans(n_clusters=k, max_iter=500)
            kmeans.fit_predict(X_train_transformed)
            inertia_values.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_train_transformed, kmeans.labels_))

        title_dic = {'fontsize': 7, 'fontweight':'bold'}

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(6, 2))
        ax1.set_xlabel("K", title_dic)
        ax1.set_title("Inertia vs K - DR dataset", title_dic)
        ax1.set_ylabel("Inertia", title_dic)
        ax1.tick_params(axis="x", labelsize=7)
        ax1.tick_params(axis="y", labelsize=7)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.plot(k_values, inertia_values, linewidth=2)
        ax1.legend(loc='best', fontsize=6)
        ax1.grid()

        ax2.set_xlabel("K", title_dic)
        ax2.set_title("Silhouette Score vs K - DR dataset", title_dic)
        ax2.set_ylabel("Silhouette Score", title_dic)
        ax2.tick_params(axis="x", labelsize=7)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.plot(k_values, silhouette_scores, linewidth=2)
        ax2.legend(loc='best', fontsize=6)
        ax2.grid()

        plt.tight_layout()

        path = os.path.join(OUTPUT)
        filename = "test.png"
        filename = os.path.join(path, filename)
        plt.savefig(filename)

        plt.close()
        kmeans = KMeans(3, max_iter=500)
        visualizer = InterclusterDistance(kmeans)
        visualizer.fit(X_train_transformed)
        visualizer.show()


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

    sys.stdout.flush()
    sys.stdout.close()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Args for experiment')
    parser.add_argument('--dataset1', metavar='path', required=False, default="diabetic.csv",
                        help='The path to dataset1')
    parser.add_argument('--dataset2', metavar='path', required=False, default="earthquake_processed.csv",
                        help='The path to dataset2')
    args = parser.parse_args()

    main(dataset1=args.dataset1,
         dataset2=args.dataset2)