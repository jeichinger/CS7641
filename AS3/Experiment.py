import pandas as pd
import numpy as np
import os
import DataPreprocessor
import PlottingUtils
import sys
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score

DATA_PATH = os.getcwd() + "/Data"
LOG_PATH = os.getcwd() + "/Logs"
OUTPUT = os.getcwd() + "/Output"

def main(dataset2="diabetic.csv", dataset1="earthquake_processed.csv", run_part_1=True):

    #sys.stdout = open(os.path.join(LOG_PATH, 'log' + time.strftime("%Y%m%d-%H%M%S") + ".txt"), 'w+')

    if dataset1 != "":

        print("Loading dataset " + dataset1, flush=True)

        # Load the data.
        dataset_csv_path = os.path.join(DATA_PATH, dataset1)
        dataset = pd.read_csv(dataset_csv_path)

        X = dataset.drop("class", axis=1)
        y = dataset["class"].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        y_train = y_train.tolist()
        y_test = y_test.tolist()

        pipe = DataPreprocessor.preprocess_data(X_train)
        X_train_transformed = pipe.fit_transform(X_train)

        if run_part_1:

            PlottingUtils.generate_pair_plot("Feature Pair Plot - True Labels - DR Dataset", X_train_transformed,
                                             np.array(y_train), columns=dataset.columns,
                                             x_labels=dataset.columns.tolist()[:-1],
                                             y_labels=dataset.columns.tolist()[:-1])

            k_values = np.arange(2, 10, 1)

            PlottingUtils.plot_k_means_scores(k_values, X_train_transformed, "Normalized Scores of Various Metrics vs K - DR Dataset")

            # By inspection, 3 was the best number of clusters.
            kmeans = KMeans(n_clusters=3, max_iter=500)
            kmeans.fit_predict(X_train_transformed)

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(kmeans.labels_, y_train)
            print("Scores for DR Dataset")
            print("K Means")
            print("homogeneity:" + str(homogeneity))
            print("completeness:" + str(completeness))
            print("v measure:" + str(v_measure))
            print("Adjusted mutual info score:" + str(adjusted_mutual_info_score(y_train, kmeans.labels_)))
            print()

            PlottingUtils.generate_pair_plot("Feature Pair Plot - KMeans - DR Dataset", X_train_transformed, kmeans.labels_, columns=dataset.columns,x_labels=dataset.columns.tolist()[:-1],
                                             y_labels=dataset.columns.tolist()[:-1])

            k_values = np.arange(2, 15, 1)
            PlottingUtils.plot_gmm_scores(k_values, X_train_transformed, "EM - DR Dataset")

            # By inspection, 4 clusters were best.
            gmm = GaussianMixture(4, max_iter=500, n_init=10)
            gmm.fit(X_train_transformed)

            labels = np.array(gmm.predict(X_train_transformed))

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels, y_train)
            print("EM")
            print("homogeneity:" + str(homogeneity))
            print("completeness:" + str(completeness))
            print("v measure:" + str(v_measure))
            print("Adjusted mutual info score:" + str(adjusted_mutual_info_score(y_train, labels)))
            print()

            PlottingUtils.generate_pair_plot("Feature Pair Plot - EM - DR Dataset", X_train_transformed, labels, columns=dataset.columns, x_labels=dataset.columns.tolist()[:-1],
                                             y_labels=dataset.columns.tolist()[:-1])


    if dataset2 != "":

        print("Loading dataset " + dataset2)

        # Load the data.
        dataset_csv_path = os.path.join(DATA_PATH, dataset2)
        dataset = pd.read_csv(dataset_csv_path)

        X = dataset.drop("class", axis=1)
        y = dataset["class"].copy()

        numeric_features = list(X.select_dtypes(include=np.number))
        cat_features = list(X.select_dtypes(exclude=np.number))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        pipe = DataPreprocessor.preprocess_data(X_train)
        X_train_transformed = pipe.fit_transform(X_train)

        enc_cat_features = pipe.named_transformers_['cat'].get_feature_names()
        labels = np.concatenate([numeric_features, enc_cat_features])
        transformed_df_columns = pd.DataFrame(pipe.transform(X_train), columns=labels).columns.tolist()
        transformed_df_columns.append("class")

        if run_part_1:

            k_values = np.arange(2, 100, 1)

            #PlottingUtils.plot_k_means_scores(k_values, X_train_transformed, "Normalized Scores of Various Metrics vs K - Earthquake Dataset")

            kmeans = KMeans(n_clusters=21, max_iter=500)
            kmeans.fit_predict(X_train_transformed)

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(kmeans.labels_, y_train)
            print("Scores for Earthquake Dataset")
            print("K Means")
            print("homogeneity:" + str(homogeneity))
            print("completeness:" + str(completeness))
            print("v measure:" + str(v_measure))
            print("Adjusted mutual info score:" + str(adjusted_mutual_info_score(y_train, kmeans.labels_)))
            print()

            #PlottingUtils.generate_tsne("TSNE Visualization of K-Means Clusters - Earthquake Dataset", X_train_transformed, kmeans.labels_)

            k_values = np.arange(2, 50, 1)
            #PlottingUtils.plot_gmm_scores(k_values, X_train_transformed, "BIC & AIC Scores EM - Earthquake Dataset")

            gmm = GaussianMixture(17, max_iter=500, n_init=10)
            gmm.fit(X_train_transformed)

            labels = np.array(gmm.predict(X_train_transformed))

            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels, y_train)

            print("EM")
            print("homogeneity:" + str(homogeneity))
            print("completeness:" + str(completeness))
            print("v measure:" + str(v_measure))
            print("Adjusted mutual info score:" + str(adjusted_mutual_info_score(y_train, labels)))

            PlottingUtils.generate_tsne("TSNE Visualization of EM Clusters - Earthquake Dataset", X_train_transformed, labels)

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