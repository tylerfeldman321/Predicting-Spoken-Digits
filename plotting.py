import matplotlib.pyplot as plt

from parse_data import load_data, get_train_data
from constants import *
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
from find_gmm_parameters import *
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import cm


def plot_mffc(digit):
    """
    :param digit: Matrix of dimensions _ x NUM_MFCC
    :return: Plots each mffc coefficient separately as a function of the time window index
    """

    num_time_windows = digit.shape[0]
    num_mfcc_coeffs = digit.shape[1]
    time_windows = range(num_time_windows)

    plt.figure()
    for i in range(num_mfcc_coeffs):
        mfcc_values = digit[:, i]
        plt.plot(time_windows, mfcc_values, label=f'Coefficient {i+1}')
    plt.xlabel('Time Window Index')
    plt.ylabel('MFCC Values')
    plt.legend(loc='best')
    plt.title('MFCC Values vs Time Window Index')
    plt.grid(True)
    plt.show()


def plot_mfcc_example_digit():
    train_data, train_labels = load_data(TRAIN_FILE)
    plot_mffc(train_data[0])


def plot_clusters_2d(mffc_frames, cluster_labels, gmm_parameters, show_mean_and_cov=False, plot_padding_value=0.5):
    n_clusters = max(list(set(cluster_labels))) + 1

    fig, ax = plt.subplots(1, 1)
    scatter = ax.scatter(mffc_frames[:, 0], mffc_frames[:, 1], c=cluster_labels, label=cluster_labels)
    ax.set_title('All Clusters')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    lp = lambda i: ax.plot([], color=scatter.cmap(scatter.norm(i)), ms=np.sqrt(20), mec="none",
                            label="Cluster {:g}".format(i), ls="", marker="o")[0]
    handles = [lp(i) for i in np.unique(cluster_labels)]
    ax.legend(handles=handles, loc='best')

    for i in range(n_clusters):
        cluster_frames = mffc_frames[cluster_labels == i]
        # ax.scatter(cluster_frames[:, 0], cluster_frames[:, 1], c=cluster_labels[cluster_labels == i])
        cluster_mean = gmm_parameters[i][0]
        cluster_cov = gmm_parameters[i][1]

        if show_mean_and_cov:
            ax.plot(cluster_mean[0], cluster_mean[1], 'o', label=f'Mean of Cluster {i}')
            N = 200
            X = np.linspace(np.min(cluster_frames[:, 0].flatten())-plot_padding_value, np.max(cluster_frames[:, 0].flatten())+plot_padding_value, N)
            Y = np.linspace(np.min(cluster_frames[:, 1].flatten())-plot_padding_value, np.max(cluster_frames[:, 1].flatten())+plot_padding_value, N)
            X, Y = np.meshgrid(X, Y)
            pos = np.dstack((X, Y))
            rv = multivariate_normal(cluster_mean, cluster_cov)
            Z = rv.pdf(pos)
            ax.contour(X, Y, Z)
    plt.show()


def plot_clusters_tsne(mffc_frames, cluster_labels, gmm_parameters, covariance_type, plot_padding_value=0.5):
    """

    :param mffc_frames: Frames for a digit
    :param cluster_labels: Labels for those frames, coming from kmeans
    :param gmm_parameters: Gaussian mixture model parameters for each kmeans cluster
    :param plot_padding_value: How much to pad the resulting 2D plot
    :return: Plots the clusters, their means, and their covariances after mapping the higher dimensional data to 2 dimensions
    """
    num_dimensions = len(mffc_frames[0])

    if not num_dimensions == 2:
        print(f'Mapping the data from {num_dimensions} to 2 dimensions')
        mffc_frames = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(mffc_frames)
        gmm_parameters = []
        n_clusters = max(list(set(cluster_labels))) + 1
        for i in range(n_clusters):
            cluster_frames = mffc_frames[cluster_labels == i]
            cluster_gmm_component_parameters = get_gmm_component_parameters(cluster_frames, len(mffc_frames), covariance_type)
            gmm_parameters.append(cluster_gmm_component_parameters)
    else:
        print("Don't need to transform the data, so just plotting in 2d")

    plot_clusters_2d(mffc_frames, cluster_labels=cluster_labels, gmm_parameters=gmm_parameters, plot_padding_value=plot_padding_value)


def kmeans_and_plot_with_tsne_for_digit(data, digit_value, covariance_type, n_clusters=4):
    assert(0 <= digit_value < NUM_DIGITS)

    # Get the clusters for a specific digit, and then plot them in 2D space using t-SNE
    mfcc_frames_list, cluster_labels_list, gaussian_parameters_list = find_gmm_component_parameters_kmeans(data=data, covariance_type=covariance_type, n_clusters=n_clusters)
    plot_clusters_tsne(mfcc_frames_list[digit_value], cluster_labels_list[digit_value], gaussian_parameters_list[digit_value], covariance_type)


if __name__ == '__main__':
    coeffs = ALL_COEFFS
    covariance_type = CovarianceType.FULL
    kmeans_and_plot_with_tsne_for_digit(get_train_data(coeffs=coeffs), digit_value=0, covariance_type=covariance_type, n_clusters=8)
    # plot_mfcc_example_digit()
