from functools import reduce
from model import DigitPredictor
from constants import *
from tqdm import tqdm


def powerset(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result],
                  lst, [[]])


def test_mfcc_combinations():
    accuracies = {}
    coeffs_list = powerset(ALL_COEFFS)
    coeffs_list = coeffs_list[1000::15]
    covariance_types = [CovarianceType.FULL]
    n_components_list = np.arange(3,7)

    print(f'Number of models to train: {len(n_components_list) * len(coeffs_list) * len(covariance_types)}')

    best_accuracy = 0
    best = ''

    for coeffs in coeffs_list:
        if len(coeffs) == 0:
            continue
        dp = DigitPredictor(coeffs=coeffs)
        for covariance_type in covariance_types:
            for n_components in n_components_list:
                model_name = f'{coeffs}-{covariance_type}-{n_components}'
                print(f'-------------RUNNING {model_name}-------------')
                dp.get_gmm_list(gmm_parameter_method=GMMParameterMethod.EM,
                                      covariance_type=covariance_type,
                                      n_components=n_components)
                accuracy = dp.evaluate(plot_confusion_matrix=False)
                accuracies[model_name] = accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best = model_name
                print(f'Best: {best}, {best_accuracy}')
    print(accuracies)

    # Best: [2, 3, 4, 5, 7, 8, 9, 10]-CovarianceType.FULL-6, 0.8959090909090909
    # Best: [2, 3, 4, 6, 7, 8, 9, 10, 11]-CovarianceType.FULL-6, 0.9040909090909091


def test_separating_gender():
    coeffs = ALL_COEFFS
    dp = DigitPredictor(coeffs)
    covs = [CovarianceType.FULL, CovarianceType.DIAG, CovarianceType.TIED, CovarianceType.SPHERICAL]
    gmm_method = GMMParameterMethod.EM
    iters = 10

    accuracies_not_separate = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(gmm_method, cov, 5, separate_genders=False)
            accuracy_cov += dp.evaluate(plot_confusion_matrix=False)
        accuracy_cov /= iters
        accuracies_not_separate.append(accuracy_cov)

    print('Moving onto separate gender model')
    accuracies_separate = []
    for cov in covs:
        accuracy_cov = 0
        for i in range(iters):
            print(f'Cov: {cov}, iteration: {i}')
            dp.get_gmm_list(gmm_method, cov, 5, separate_genders=True)
            accuracy_cov += dp.evaluate(plot_confusion_matrix=False)
        accuracy_cov /= iters
        accuracies_separate.append(accuracy_cov)

    print(f'Accuracies for Not Separating Based on Gender: {accuracies_not_separate}')
    print(f'Accuracies for Separating Based on Gender: {accuracies_separate}')


if __name__ == '__main__':
    test_separating_gender()
