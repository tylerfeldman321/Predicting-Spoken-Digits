from functools import reduce
from model import DigitPredictor
from constants import *
from tqdm import tqdm


def powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])


def test_mfcc_combinations(powerset=False, stride=100):
    accuracies = {}

    coeffs_list = None
    if powerset:
        # Random sets of coefficients
        coeffs_list = powerset(ALL_COEFFS)
        coeffs_list = coeffs_list[0::stride]
    else:
        # Decreasing number of coefficients
        coeffs_list = [[0,1,2,3,4,5,6,7,8,9,10,11,12],
                       [0,1,2,3,4,5,6,7,8,9,10,11],
                       [0,1,2,3,4,5,6,7,8,9,10],
                       [0,1,2,3,4,5,6,7,8,9],
                       [0,1,2,3,4,5,6,7,8],
                       [0,1,2,3,4,5,6,7],
                       [0,1,2,3,4,5,6],
                       [0,1,2,3,4,5],
                       [0,1,2,3,4],
                       [0,1,2,3],
                       [0,1,2],
                       [0,1],
                       [0]]

    covariance_types = [CovarianceType.DIAG]
    n_components = 7

    print(f'Number of models to train: {len(coeffs_list) * len(covariance_types)}')

    best_accuracy = 0
    best = ''

    for coeffs in coeffs_list:
        if len(coeffs) == 0:
            continue
        dp = DigitPredictor(coeffs=coeffs)
        for covariance_type in covariance_types:
            avg_accuracy = 0
            for i in range(10):
                model_name = f'{coeffs}-{covariance_type}-{n_components}-{i}'
                print(f'-------------RUNNING {model_name}-------------')
                dp.get_gmm_list(gmm_parameter_method=GMMParameterMethod.EM,
                                      covariance_type=covariance_type,
                                      n_components=n_components)
                accuracy = dp.evaluate(plot_confusion_matrix=False)
                avg_accuracy += accuracy
            avg_accuracy /= 10
            accuracies[model_name] = avg_accuracy
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best = model_name
            print(f'Best: {best}, {best_accuracy}')
    print(accuracies)


def test_separating_gender():
    coeffs = ALL_COEFFS
    dp = DigitPredictor(coeffs)
    covs = [CovarianceType.FULL, CovarianceType.DIAG, CovarianceType.TIED, CovarianceType.SPHERICAL]
    n_components = 7
    gmm_method = GMMParameterMethod.EM
    iters = 10

    accuracies_not_separate = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(gmm_method, cov, n_components, separate_genders=False)
            accuracy_cov += dp.evaluate(plot_confusion_matrix=False)
        accuracy_cov /= iters
        accuracies_not_separate.append(accuracy_cov)

    print('Moving onto separate gender model')
    accuracies_separate = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(gmm_method, cov, n_components, separate_genders=True)
            accuracy_cov += dp.evaluate(plot_confusion_matrix=True)
        accuracy_cov /= iters
        accuracies_separate.append(accuracy_cov)

    print(f'Accuracies for Not Separating Based on Gender: {accuracies_not_separate}')
    print(f'Accuracies for Separating Based on Gender: {accuracies_separate}')


def test_different_covariances_and_kmeans_vs_EM():
    covs = [CovarianceType.FULL, CovarianceType.DIAG, CovarianceType.TIED, CovarianceType.SPHERICAL]
    GMM_methods = [GMMParameterMethod.EM, GMMParameterMethod.KMEANS]
    dp = DigitPredictor()
    n_components = 7
    iters = 10

    accuracies_em = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'GMM Method: {GMM_methods[0]}, Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(GMM_methods[0], cov, n_components, separate_genders=False)
            accuracy_cov += dp.evaluate()
        accuracy_cov /= iters
        accuracies_em.append(accuracy_cov)

    accuracies_kmeans = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'GMM Method: {GMM_methods[1]}, Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(GMM_methods[1], cov, n_components, separate_genders=False)
            accuracy_cov += dp.evaluate()
        accuracy_cov /= iters
        accuracies_kmeans.append(accuracy_cov)

    print(accuracies_em)
    print(accuracies_kmeans)


if __name__ == '__main__':
    test_mfcc_combinations()
    test_separating_gender()
    test_different_covariances_and_kmeans_vs_EM()
