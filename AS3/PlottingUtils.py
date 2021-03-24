from sklearn.cluster import KMeans
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import InterclusterDistance
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns
from yellowbrick.style.colors import resolve_colors
from yellowbrick.text import TSNEVisualizer

DATA_PATH = os.getcwd() + "/Data"
LOG_PATH = os.getcwd() + "/Logs"
OUTPUT = os.getcwd() + "/Output"

def plot_k_means_scores(k_range, X_train_transformed, title):
    inertia_values = []
    silhouette_scores = []
    calinski_scores = []
    davies_score = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, max_iter=500)
        kmeans.fit_predict(X_train_transformed)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_train_transformed, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X_train_transformed, kmeans.labels_))
        davies_score.append(davies_bouldin_score(X_train_transformed, kmeans.labels_))

    scaler = MinMaxScaler()
    inertia_values = scaler.fit_transform(np.asarray(inertia_values).reshape(-1, 1))
    silhouette_scores = scaler.fit_transform(np.asarray(silhouette_scores).reshape(-1, 1))
    calinski_scores = scaler.fit_transform(np.asarray(calinski_scores).reshape(-1, 1))
    davies_score = scaler.fit_transform(np.asarray(davies_score).reshape(-1, 1))

    title_dic = {'fontsize': 7, 'fontweight': 'bold'}

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 2))
    ax1.set_xlabel("K", title_dic)
    ax1.set_title(title, title_dic)
    ax1.set_ylabel("Normalized Score/Index", title_dic)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.plot(k_range, inertia_values, label="Inertia", linewidth=2)
    ax1.grid()

    ax1.plot(k_range, silhouette_scores, label="Silhouette Score", linewidth=2)
    ax1.plot(k_range, calinski_scores, label="Calinski-Harabasz Index", linewidth=2)
    ax1.plot(k_range, davies_score, label="Davies-Bouldin Index", linewidth=2)

    ax1.legend(loc='best', fontsize=6)

    plt.tight_layout()
    plt.grid()

    path = os.path.join(OUTPUT)
    filename = title + ".png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)
    plt.close()

def plot_gmm_scores(k_range, X_train_transformed, title):
    bic_scores = []
    aic_scores = []
    for k in k_range:
        gmm = GaussianMixture(k, max_iter=500, n_init=10)
        gmm.fit(X_train_transformed)
        bic_scores.append(gmm.bic(X_train_transformed))
        aic_scores.append(gmm.aic(X_train_transformed))

    title_dic = {'fontsize': 7, 'fontweight': 'bold'}

    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 2))
    ax1.set_xlabel("K", title_dic)
    ax1.set_title(title, title_dic)
    ax1.set_ylabel("Score", title_dic)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.plot(k_range, bic_scores, label="BIC", linewidth=2)
    ax1.grid()

    ax1.plot(k_range, aic_scores, label="AIC", linewidth=2)
    ax1.legend(loc='best', fontsize=6)

    plt.tight_layout()
    plt.grid()

    path = os.path.join(OUTPUT)
    filename = title + ".png"
    filename = os.path.join(path, filename)
    plt.savefig(filename)
    plt.close()

def generate_pair_plot(title, X, labels, columns, x_labels, y_labels):

    labels = labels.reshape((-1,1))
    data = np.hstack((X, labels))
    df = pd.DataFrame(data, columns=columns)

    palette = sns.color_palette("bright")
    #sns.set_palette(palette)

    plot = sns.pairplot(df, hue="class", corner=True, diag_kind="hist")
    path = os.path.join(OUTPUT)
    filename = title
    filename = os.path.join(path, filename)
    plot.savefig(filename)

def generate_tsne(title, X, labels):

    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 2))
    title_dic = {'fontsize': 7, 'fontweight': 'bold'}

    colors = resolve_colors(11, 'Spectral_r')
    colors2 = resolve_colors(10, 'BrBG_r')
    tsne = TSNEVisualizer(ax1, colors=colors + colors2,decompose=None)
    tsne.fit(X, labels)
    tsne.finalize()
    ax1 = tsne.ax
    ax1.set_title(title, title_dic)

    path = os.path.join(OUTPUT)
    filename = title
    filename = os.path.join(path, filename)
    plt.savefig(filename)