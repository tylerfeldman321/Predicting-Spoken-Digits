from constants import *
import numpy as np


def load_data(filepath, coeffs):
    """
    :param filepath: Path for the file to read, either the TRAIN_FILE or TEST_FILE
    :return:
    Organizes the data in the input file into a list of matrices. Each matrix is the data for a spoken digit,
     where the columns are the coefficients and each row is the index of the time window
    """
    mask = convert_list_to_mask(coeffs)

    digits = []
    labels = []
    current_digit = []
    with open(filepath, 'r') as f:
        # For each block, append the items into one long list and then reshape to be (-1, 13) and append to digits
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            if line.isspace() or line == "\n":
                digits = add_digit(digits, current_digit)
                current_digit = []
            else:
                mfcc = list(map(float, line.split(' ')))
                mfcc = np.asarray(mfcc)
                mfcc_filtered = mfcc[mask]
                current_digit.append(mfcc_filtered)
        if len(current_digit):
            digits = add_digit(digits, current_digit)

    num_entries_per_digit = int(len(digits) / 10)  # Should be 660 for train, 220 for test
    for digit_value in range(NUM_DIGITS):
        for i in range(num_entries_per_digit):
            labels.append(digit_value)

    return np.asarray(digits, dtype=object), np.asarray(labels)


def add_digit(digits, current_digit):
    """
    :param digits: List of digits, which are each a numpy array
    :param current_digit: Current digit, which is a list of the coefficients for the digit
    :return:
    Formats and reshapes the current_digit list as a numpy array and appends it to digits and returns digits
    """
    digit_matrix = np.asarray(current_digit)
    digit_matrix = np.reshape(digit_matrix, (-1, len(current_digit[0])))
    digits.append(digit_matrix)
    return digits


def get_train_data(coeffs=ALL_COEFFS):
    return load_data(TRAIN_FILE, coeffs)


def get_test_data(coeffs=ALL_COEFFS):
    return load_data(TEST_FILE, coeffs)


def convert_list_to_mask(list_of_indices):
    mask = [False] * NUM_MFCC
    for index in list_of_indices:
        assert (0 <= index < NUM_MFCC)
        mask[index] = True
    return mask


if __name__ == "__main__":
    # train_digits, train_labels = get_train_data()
    coeffs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    test_digits, test_labels = get_test_data()
    assert (len(test_digits[0][0]) == len(coeffs))
