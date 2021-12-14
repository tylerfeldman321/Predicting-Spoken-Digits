import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from constants import *
from parse_data import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from typing import List, Union
import time


class DigitPredictor:
    def __init__(self, coeffs=ALL_COEFFS):
        self.coeffs = coeffs
        self.n_features = len(coeffs)

        print('Loading Training Data...')
        self.train_digits, self.train_labels = get_train_data(coeffs)
        print('Loading Testing Data...')
        self.test_digits, self.test_labels = get_test_data(coeffs)

        self.test_predictions = []
        self.test_accuracy = None

        self.mfcc_frames_list = None
        self.cluster_labels_list = None

        self.gmm_list = []
        self.separate_genders = None

    def get_gmm_list(self, gmm_parameter_method: GMMParameterMethod, covariance_type: CovarianceType, n_components: Union[List, int], separate_genders: bool = False):
        self.test_predictions = []  # If running with new gmm parameters, then we want to reset the test predictions
        self.gmm_list = []

        self.separate_genders = separate_genders

        assert (type(gmm_parameter_method) == GMMParameterMethod), 'gmm_parameter_method must be an object with class GMMParameterMethod'
        assert (type(covariance_type) == CovarianceType), 'covariance_type must be an object with class CovarianceType'

        self.mfcc_frames_list = []  # MFCC frames for each digit
        self.cluster_labels_list = []  # Cluster labels for each digit

        if type(n_components) == int:
            n_components = [n_components] * NUM_DIGITS
        assert(len(n_components) == NUM_DIGITS)

        # Find GMM parameters for each digit
        for target_digit_value in tqdm(range(NUM_DIGITS), ascii=False,
                                       desc=f'Finding GMM Parameters for Each Digit, with {gmm_parameter_method}, {covariance_type}'):
            target_digit_indices = np.where(self.train_labels == target_digit_value, True, False)
            target_digits = self.train_digits[target_digit_indices]  # Get the digit blocks for target digit
            assert (len(target_digits) == sum(target_digit_indices))

            # Get all of the frames for the blocks for target digit
            mfcc_frames = []
            for digit_block in target_digits:
                for mfcc_coeffs in digit_block:
                    mfcc_frames.append(mfcc_coeffs)
            mfcc_frames = np.asarray(mfcc_frames)

            if not separate_genders:
                self._get_gmm(mfcc_frames, gmm_parameter_method, covariance_type, n_components[target_digit_value])
            else:
                male_frames = mfcc_frames[:len(mfcc_frames)//2]
                female_frames = mfcc_frames[len(mfcc_frames)//2:]
                self._get_gmm(male_frames, gmm_parameter_method, covariance_type, n_components[target_digit_value])
                self._get_gmm(female_frames, gmm_parameter_method, covariance_type, n_components[target_digit_value])

    def _get_gmm(self, mfcc_frames, gmm_parameter_method, covariance_type, n_components):
        gmm = None

        if gmm_parameter_method == GMMParameterMethod.KMEANS:
            # Apply clustering to the data
            kmeans = KMeans(n_clusters=n_components).fit(mfcc_frames)
            cluster_labels = kmeans.labels_
            self.cluster_labels_list.append(cluster_labels)

            cluster_weights = []
            cluster_means = []
            cluster_covariances = []
            cluster_precisions = []

            if covariance_type == CovarianceType.TIED:
                mfcc_frames_demeaned = mfcc_frames.copy()

            for i in range(n_components):
                cluster_frames = mfcc_frames[cluster_labels == i]
                cluster_mean = np.mean(cluster_frames, axis=0, dtype=np.float64)

                # Calculate covariance
                if covariance_type == CovarianceType.FULL:
                    cluster_cov = np.cov(cluster_frames, rowvar=False)  # Each row represents a single observation, each column is a variable
                elif covariance_type == CovarianceType.DIAG:
                    cluster_cov = np.zeros((cluster_mean.shape[0], cluster_mean.shape[0]))
                    for j in range(cluster_mean.shape[0]):
                        cluster_cov[j, j] = sum((val - cluster_mean[j]) ** 2 for val in cluster_frames[:, j]) / cluster_frames.shape[0]
                elif covariance_type == CovarianceType.SPHERICAL:
                    demeaned_cluster_frames = cluster_frames - cluster_mean
                    var = np.var(demeaned_cluster_frames.flatten())
                    cluster_cov = np.identity(cluster_mean.shape[0]) * var
                elif covariance_type == CovarianceType.TIED:
                    mfcc_frames_demeaned[cluster_labels == i] = mfcc_frames_demeaned[cluster_labels == i] - cluster_mean

                # Calculate weight of the cluster
                cluster_weight = len(cluster_frames) / len(mfcc_frames)

                cluster_weights.append(cluster_weight)
                cluster_means.append(cluster_mean)
                if not covariance_type == CovarianceType.TIED:
                    cluster_covariances.append(cluster_cov)
                    cluster_precisions.append(np.linalg.inv(cluster_cov))

            if covariance_type == CovarianceType.TIED:
                cluster_cov = np.cov(mfcc_frames_demeaned, rowvar=False)
                cluster_covariances = [cluster_cov] * n_components

            cluster_weights, cluster_means, cluster_covariances, cluster_precisions = np.asarray(cluster_weights), \
                                                                                      np.asarray(cluster_means), \
                                                                                      np.asarray(cluster_covariances), \
                                                                                      np.asarray(cluster_precisions)

            # Create GMM by manually setting all of its parameters
            gmm = GaussianMixture(n_components=n_components, covariance_type='full')
            gmm.weights_ = cluster_weights
            gmm.covariances_ = cluster_covariances
            gmm.precisions_ = cluster_precisions
            gmm.means_ = cluster_means
            gmm.precisions_cholesky_ = _compute_precision_cholesky(cluster_covariances, "full")

        elif gmm_parameter_method == GMMParameterMethod.EM:
            # Initialize GMM and fit it to the data for the digit using expectation maximization
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type.value).fit(mfcc_frames)
            # gmm = BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type.value, max_iter=200, n_init=3).fit(mfcc_frames)

        self.gmm_list.append(gmm)
        self.mfcc_frames_list.append(mfcc_frames)

    def predict(self):
        assert(len(self.gmm_list)), 'You must call get_gmm_list() before calling predict'
        if len(self.test_predictions) == 0:
            self.test_predictions = np.zeros(self.test_labels.shape)
            for i in tqdm(range(len(self.test_digits)), ascii=False, desc='Generating Predictions'):
                digit_frames = self.test_digits[i]
                digit_likelihoods = []
                for digit_num in range(NUM_DIGITS):
                    if not self.separate_genders:
                        gmm = self.gmm_list[digit_num]
                        score = gmm.score(digit_frames)
                    else:
                        gmm_male, gmm_female = self.gmm_list[digit_num*2], self.gmm_list[(digit_num * 2) + 1]
                        score = max(gmm_male.score(digit_frames), gmm_female.score(digit_frames))
                    digit_likelihoods.append(score)
                self.test_predictions[i] = np.argmax(np.asarray(digit_likelihoods))
        return self.test_predictions

    def evaluate(self, plot_confusion_matrix=False):
        predictions = self.predict()
        self.test_accuracy = (predictions == self.test_labels).sum() / len(self.test_labels)
        print(f'Test Accuracy: {self.test_accuracy}')
        if plot_confusion_matrix:
            ConfusionMatrixDisplay.from_predictions(self.test_labels, predictions, normalize='true', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.show()
        return self.test_accuracy
