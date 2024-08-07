import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_regression_metrics(y_true, y_pred, X_test):

    # create a dictionary to store the results
    results = {}

    # Compute the metrics and store them in the dictionary
    results['mse'] = mean_squared_error(y_true, y_pred)
    results['rmse'] = np.sqrt(results['mse'])
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['r2'] = r2_score(y_true, y_pred)
    results['adj_r2'] = 1 - (1 - results['r2']) * (len(y_true) - 1) / (len(y_true) - X_test.shape[1] - 1)

    return results
