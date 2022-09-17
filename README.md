# Predicting-Spoken-Digits
Duke ECE 480 Applied Probability for Statistical Learning project to predict spoken digits in Arabic using a Gaussian Mixture Model (GMM). For detailed information about this project, see the included pdf.

## Modeling Approach
Each digit can be represented as a unique collection of phonemes, which are sounds that make up a noise. The model we are using is the Gaussian Mixture Model (GMM), where each Gaussian component represents one of these phonemes. We can create one GMM to model each digit. To find the parameters for each GMM, the two approaches implemented are expectation-maximization and K-Means clustering followed by manual calculation of the Gaussian parameters for each digit. After finding these ten GMM's, we can use maximum likelihood classification to predict the class for a new digit. There are many additional modeling options experimented with, such as creating two models for each digit: one for the male speakers, and one for the female speakers.

## Dataset
The dataset is the Spoken Arabic Digit Data Set, available at https://archive.ics.uci.edu/ml/datasets/Spoken+Arabic+Digit. It contains Mel-Frequency Cepstrum Coefficients (MFCC) values for 8800 different audio samples of the spoken Arabic digits 0-9. For more information, see the data set description on the website.

## Structure
`model.py`: Contains DigitPredictor class that loads data when initialized. `get_gmm_list()` will find the GMM parameters either using K-Means clustering or expectation-maximization.

`parse_data.py`: Contains the code to load the train and test set data.

`plotting.py`: Contains code to generate various plots, such as the change in MFCC values over time for a single spoken digit sample.

`experiments.py`: Contains code to run various experiments to test the performance of modeling options.

`constants.py`: Contains constants and paths to the data files.
