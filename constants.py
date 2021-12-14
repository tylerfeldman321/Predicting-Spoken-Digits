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


