import pandas as pd

from metrics.regression import compute_regression_metrics
from utils.plot import create_scatter_plot
from preprocessing.preprocess import get_data_labels_ts
from preprocessing.preprocess import exec_normalization
from utils.print import print_regression_results
from sklearn.model_selection import train_test_split


def perform_regression(data, scaler, reg, regression_type, print_results=False):
    # Split data into features, labels and timestamps
    data, labels, ts = get_data_labels_ts(data, regression_type)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, ts, test_size=0.5)#, random_state=123)

    # Data normalization
    if scaler is not None:
        X_train = exec_normalization(X_train, scaler)
        X_test = exec_normalization(X_test, scaler)

    # Fit the regressor on the training data
    reg.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = reg.predict(X_test)

    # Compute the metrics
    results = compute_regression_metrics(y_test, y_pred, X_test)

    # Create a scatter plot
    plot = create_scatter_plot(y_pred, y_test, print_plot=False)

    if print_results:
        print_regression_results(results)

    return results, plot
