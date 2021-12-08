import os
import numpy as np
from enum import Enum

TRAIN_FILE = os.path.join('Data/Train_Arabic_Digit.txt')
TEST_FILE = os.path.join('Data/Test_Arabic_Digit.txt')

NUM_MFCC = 13
ALL_COEFFS = np.arange(0, NUM_MFCC, 1)

NUM_DIGITS = 10

# Info about Train_Arabic_Digit.txt
NUM_TRAIN_BLOCKS = 660 * NUM_DIGITS

# Info about Test_Arabic_Digit.txt
NUM_TEST_BLOCKS = 220 * NUM_DIGITS


class GMMParameterMethod(Enum):
    KMEANS = 0
    EM = 1


class CovarianceType(Enum):
    FULL = 'full'
    TIED = 'tied'
    DIAG = 'diag'
    SPHERICAL = 'spherical'


class CovarianceConstraints(Enum):
    SPHERICAL = 0  # Only covariance diagonal. ALl diagonal entries are the same, so need to estimate 1 parameter for the covariance
    INDEPENDENT = 1  # Only covariance diagonal. Entries can be different so need to estimate D parameters for covariance
    FULL = 2  # Full covariance matrix, need to estimate (D^2 / 2 + D/2) parameters


class CovarianceRelationships(Enum):
    SAME = 0  # Means the covariance matrices will be equal for all mixture components / clusters
    UNIQUE = 1  # Means the covariance matrices for different mixture components will not be restricted to be equal
