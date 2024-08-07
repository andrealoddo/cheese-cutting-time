import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from metrics.classification import compute_classification_metrics, compute_classification_metrics_average
from preprocessing.preprocess import get_data_labels_ts, exec_normalization
import copy


def exec_anomaly(clf, X_train, y_train, X_test, y_test, task):
    # fit the data and tag outliers
    if clf.__class__.__name__ == "LocalOutlierFactor":
        y_pred = clf.fit_predict(X_test)
    else:
        y_pred = clf.fit(X_train).predict(X_test)

    if task == "AnomalyClassic":
        # Convert the results to the classic anomaly detection results
        y_pred = np.where(y_pred == -1, 'Positive', 'Negative')
        y_test = np.where(y_test == -1, 'Positive', 'Negative')
    else:
        # Convert the results to the classic anomaly detection results
        y_pred = np.where(y_pred == -1, 1, 0)
        y_test = np.where(y_test == -1, 1, 0)

    # Compute the metrics
    results = compute_classification_metrics(y_test, y_pred)

    return results


def perform_anomaly(data, scaler, alg, cross_val=False, loo=False, task="Anomaly"):
    # Split data into features, labels and timestamps
    data, labels, ts = get_data_labels_ts(data)

    if cross_val:
        results = perform_anomaly_cross_validation(data, labels, scaler, alg, loo, task)
    else:
        results = perform_anomaly_single_split(data, labels, scaler, alg, task)

    if cross_val:
        # perform an average of the results
        results_avg, results_std = compute_classification_metrics_average(results)
    else:
        results_avg, results_std = results, None

    return results_avg, results_std

def perform_anomaly_cross_validation(data, labels, scaler, alg, loo, task):

    if loo:
        kfold = LeaveOneOut()
        y_pred_all = []
        y_test_all = []
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=123)

    results = []

    # perform cross validation
    for train_index, test_index in kfold.split(data):
        # Get the training and test data
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        # Data normalization
        if scaler is not None:
            X_train = exec_normalization(X_train, scaler)
            X_test = exec_normalization(X_test, scaler)

        if loo:
            y_test_all.append(y_test.iloc[0])
            y_pred = exec_anomaly_loo(alg, X_train, y_train, X_test, y_test, task)
            y_pred_all.append(y_pred[0])

        # Perform anomaly detection on the current fold
        fold_results = exec_anomaly(alg, X_train, y_train, X_test, y_test, task)
        results.append(fold_results)

    if loo:
        loo_result = compute_classification_metrics(y_test_all, y_pred_all)
        results.append(loo_result)
    else:
        results = results

    return results


def perform_anomaly_classic_cross_validation(data, labels, scaler, alg, loo, task):

    if loo:
        kfold = LeaveOneOut()
        y_pred_all = []
        y_test_all = []
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=123)

    results = []

    majority_class, minority_class = labels.value_counts().index[0], labels.value_counts().index[1]

    majority_class_data = data[labels == majority_class]
    minority_class_data = data[labels == minority_class]
    majority_class_data_labels = labels[labels == majority_class]
    minority_class_data_labels = labels[labels == minority_class]

    for train_index, test_index in kfold.split(majority_class_data):
        # Split data into train and test sets.
        X_train, X_test = majority_class_data.iloc[train_index], majority_class_data.iloc[test_index]
        y_train, y_test = majority_class_data_labels.iloc[train_index], majority_class_data_labels.iloc[test_index]

        # Append the minority class data to the test set
        X_test = X_test.append(minority_class_data)
        # Append the minority class label to the test set labels
        y_test = y_test.append(minority_class_data_labels)

        # Data normalization
        if scaler is not None:
            X_train = exec_normalization(X_train, scaler)
            X_test = exec_normalization(X_test, scaler)

        if loo:
            y_test_all.append(y_test.iloc[0])
            y_pred = exec_anomaly_loo(alg, X_train, y_train, X_test, y_test, task)
            y_pred_all.append(y_pred[0])

        # Perform anomaly detection on the current fold
        fold_results = exec_anomaly(alg, X_train, majority_class_data_labels, X_test, y_test, task)
        results.append(fold_results)

    return results

def exec_anomaly_loo(clf, X_train, y_train, X_test, y_test, task):

    # fit the data and tag outliers
    if clf.__class__.__name__ == "LocalOutlierFactor":
        y_pred = clf.fit_predict(X_test)
    else:
        y_pred = clf.fit(X_train, y_train.values.ravel()).predict(X_test)

    if task == "AnomalyClassic":
        # Convert the results to the classic anomaly detection results
        y_pred = np.where(y_pred == -1, 'Negative', 'Positive')
        y_test = np.where(y_test == -1, 'Negative', 'Positive')
    else:
        # Convert the results to the classic anomaly detection results
        # -1 = outlier (no-target), 1 = inlier (target)
        y_pred = np.where(y_pred == -1, 0, 1)
        y_test = np.where(y_test == -1, 0, 1)

    return y_pred

def perform_anomaly_single_split(data, labels, scaler, alg, task):
    majority_class, minority_class = labels.value_counts().index[0], labels.value_counts().index[1]

    majority_class_data = data[labels == majority_class]
    minority_class_data = data[labels == minority_class]
    majority_class_data_labels = labels[labels == majority_class]

    # split majority class data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(majority_class_data, majority_class_data_labels,
                                                        test_size=0.5, random_state=123)

    # Append the minority class data to the test set
    X_test = X_test.append(minority_class_data)
    # Append the minority class label to the test set labels
    y_test = y_test.append(labels[labels == minority_class])

    # Data normalization
    if scaler is not None:
        X_train = exec_normalization(X_train, scaler)
        X_test = exec_normalization(X_test, scaler)

    results = exec_anomaly(alg, X_train, y_train, X_test, y_test, task)

    return results
